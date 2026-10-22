# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.context_parallel import dsa_cp


class CommonMetadata(SimpleNamespace):
    def replace(self, **kwargs):
        return CommonMetadata(**(vars(self) | kwargs))


def make_builder(cls, ratio=0):
    """Use real builder methods with CPU buffers and mocked device operators."""
    builder = cls.__new__(cls)
    builder.compressor_ratio = ratio
    builder.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=2)
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            sliding_window=4,
            num_attention_heads=8,
            index_topk=16,
            index_n_heads=8,
            index_head_dim=8,
        ),
        get_head_size=lambda: 8,
    )
    builder.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=builder.model_config,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
    )
    builder._device_metadata_enabled = False
    builder.decode_threshold = 2
    builder.num_actual_tokens = 2
    builder.num_decodes, builder.num_prefills = 1, 0
    builder.num_decode_tokens, builder.num_prefill_tokens = 2, 0
    builder.seq_lens = torch.tensor([6], dtype=torch.int32)
    builder.block_table = torch.tensor([[3, 4]], dtype=torch.int32)
    builder.slot_mapping = torch.zeros(8, dtype=torch.int32)
    builder.start_pos_prefill = torch.full((4,), -1, dtype=torch.int32)
    builder.qli_seqused_k = torch.full((4,), -1, dtype=torch.int32)
    builder.qli_cmp_residual_k = torch.full((4,), -1, dtype=torch.int32)
    builder.sas_metadata_buffer = torch.zeros(dsa_v1.DSA_METADATA_BUFFER_SIZE, dtype=torch.int32)
    builder.qli_metadata_buffer = torch.zeros_like(builder.sas_metadata_buffer)
    builder.seqused_q = torch.zeros(4, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.zeros(4, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.zeros(4, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    builder.dspark_swa_indices_buffer = torch.empty((8, 1, 8), dtype=torch.int32)
    builder.storage_block_size = 4
    builder.cache_group_key = "cache"
    builder.hadamard = None
    builder.common_ratio_to_sas_metadata = {}
    builder.metadata_cls = dsa_v1.AscendDSAMetadata
    builder._num_compressor_metadata_rows = Mock(return_value=2)
    builder.set_num_actual_tokens = lambda common: setattr(builder, "num_actual_tokens", common.num_actual_tokens)
    return builder


class TestDSAMetadataBuilders(TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(
                dsa_v1.AscendDSAMetadataBuilder,
                "_dspark_swa_indices_fast_path_eligible",
                return_value=False,
            )
        )
        stack.enter_context(patch.object(dsa_v1, "_draft_uses_sparse_flash_mla", return_value=False))
        self.builder = make_builder(dsa_v1.AscendDSAMetadataBuilder)
        self.common = CommonMetadata(
            num_reqs=1,
            num_actual_tokens=2,
            num_input_tokens=2,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=self.builder.seq_lens,
            _seq_lens_cpu=self.builder.seq_lens,
            block_table_tensor=self.builder.block_table,
            slot_mapping=torch.tensor([12, 13], dtype=torch.int32),
            positions=torch.tensor([4, 5]),
            causal=False,
            attn_state=None,
        )
        self.sas_op = Mock(
            side_effect=lambda **kw: torch.full(
                (dsa_v1.DSA_METADATA_BUFFER_SIZE,),
                2 if kw["ori_mask_mode"] == 0 else 1,
                dtype=torch.int32,
            )
        )
        self.qli_op = stack.enter_context(
            patch.object(
                torch.ops._C_ascend,
                "npu_quant_lightning_indexer_v2_metadata",
                create=True,
                return_value=torch.ones(dsa_v1.DSA_METADATA_BUFFER_SIZE, dtype=torch.int32),
            )
        )
        plan = SimpleNamespace(
            layout_kv="PA_BSND",
            get_dsa_sparse_attn_metadata_op=lambda: self.sas_op,
            get_dsa_sparse_attn_metadata_kwargs=lambda device: {},
            format_dsa_slot_mapping=lambda slots, block_size: slots,
        )
        stack.enter_context(patch.object(dsa_v1, "get_dsa_attn_kv_plan", return_value=plan))
        stack.enter_context(patch.object(dsa_v1, "_dsa_swa_only_cmp_ratio", return_value=1))
        stack.enter_context(patch.object(dsa_v1, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(dsa_v1, "is_turboquant", return_value=False))
        stack.enter_context(patch.object(dsa_v1, "get_dspark_sparse_sas_window", return_value=(5, 0)))
        stack.enter_context(patch.object(dsa_v1, "get_full_cos_and_sin_dsa", return_value=(None, None)))
        self.rope_op = stack.enter_context(patch.object(dsa_v1, "get_cos_and_sin_dsa", return_value=(None, None)))
        self.indices_op = stack.enter_context(
            patch.object(
                dsa_v1,
                "build_dspark_swa_indices",
                return_value=(torch.ones((2, 1, 8), dtype=torch.int32), None),
            )
        )
        stack.enter_context(
            patch.object(
                dsa_v1,
                "DeviceOperator",
                SimpleNamespace(
                    get_dsa_decode_cu_seqlens_ori_kv=lambda *args: None,
                    get_dsa_decode_cu_seqlens_cmp_kv=lambda *args: None,
                    get_dsa_indexer_quant_mode=lambda: 0,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                dsa_v1,
                "split_decodes_and_prefills",
                side_effect=lambda common, **kwargs: (common.num_reqs, 0, common.num_actual_tokens, 0),
            )
        )

    def build(self, builder, common, shared):
        return builder.build(0, common, common_ratio_to_sas_metadata=shared)

    def test_non_pcp_shares_results_with_independent_output_buffers(self):
        shared = {}
        common = self.common.replace(causal=True)
        builders = [make_builder(dsa_v1.AscendDSAMetadataBuilder, ratio=4) for _ in range(2)]
        first, second = [self.build(builder, common, shared).req_metadata for builder in builders]
        self.rope_op.assert_called_once()
        self.sas_op.assert_called_once()
        self.qli_op.assert_called_once()
        for name in ("sas_metadata", "qli_metadata", "qli_seqused_k", "qli_cmp_residual_k", "start_pos"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))
            self.assertNotEqual(getattr(first, name).data_ptr(), getattr(second, name).data_ptr())

    def test_sas_cache_separates_causal_and_dspark_groups(self):
        shared = {}
        builders = [make_builder(dsa_v1.AscendDSAMetadataBuilder) for _ in range(3)]
        causal = self.build(builders[0], self.common.replace(causal=True), shared).req_metadata
        draft = self.build(builders[1], self.common, shared).req_metadata
        repeated = self.build(builders[2], self.common, shared).req_metadata
        self.assertEqual(self.sas_op.call_count, 2)
        self.assertTrue(torch.all(causal.sas_metadata == 1))
        self.assertTrue(torch.all(draft.sas_metadata == 2))
        torch.testing.assert_close(draft.sas_metadata, repeated.sas_metadata)
        self.assertEqual(self.indices_op.call_count, 2)

    def test_sas_cache_separates_windows(self):
        shared = {}
        self.build(self.builder, self.common, shared)
        other = make_builder(dsa_v1.AscendDSAMetadataBuilder)
        with patch.object(dsa_v1, "get_dspark_sparse_sas_window", return_value=(7, 0)):
            self.build(other, self.common, shared)
        self.assertEqual(self.sas_op.call_count, 2)
        self.assertEqual(self.sas_op.call_args.kwargs["ori_win_left"], 7)

    def make_pcp_builder(self, global_common, local_common, *, sharded=False, full_graph=False):
        builder = make_builder(dsa_cp.AscendDSAPCPMetadataBuilder, ratio=4)
        builder._global_metadata_builder = make_builder(dsa_v1.AscendDSAMetadataBuilder, ratio=4)
        builder._is_decode_sharded = sharded
        builder._global_rope_buffers = {}
        builder._hidden_restore_idx_buffer = torch.empty(8, dtype=torch.int64)
        builder.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs = lambda: full_graph
        builder._prepare_graph_pcp_context = Mock(side_effect=lambda context, **kwargs: context)
        builder._build_global_common_attn_metadata = Mock(return_value=global_common)
        builder._build_graph_common_attn_metadata = Mock(side_effect=lambda common, actual: common)
        builder._build_local_common_attn_metadata = Mock(return_value=local_common)
        return builder

    def run_pcp_groups(self, *, prefill=True, sharded=False, full_graph=False, steps=1, empty_local=False):
        # Exercise the real PCP build and ordinary DSA builders. Only PCP layout
        # preparation and device operators are replaced with CPU fixtures.
        global_common = self.common.replace(
            causal=True,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
            positions=torch.tensor([8, 9]),
        )
        local_common = self.common.replace(causal=True)
        if empty_local:
            local_common = local_common.replace(
                num_actual_tokens=0,
                num_input_tokens=0,
                query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
                query_start_loc_cpu=torch.tensor([0, 0], dtype=torch.int32),
            )
        builders = [
            self.make_pcp_builder(
                global_common.replace(
                    block_table_tensor=torch.tensor([[10 + group, 20 + group]], dtype=torch.int32),
                    slot_mapping=torch.tensor([40 + group, 80 + group], dtype=torch.int32),
                ),
                local_common.replace(block_table_tensor=torch.tensor([[30 + group, 40 + group]], dtype=torch.int32)),
                sharded=sharded,
                full_graph=full_graph,
            )
            for group in range(2)
        ]
        context = SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([prefill]), num_reqs=1),
            hidden_restore_idx=torch.tensor([0, 1]),
        )
        runs = []
        for step in range(steps):
            shared = {}
            outputs = []
            for group, builder in enumerate(builders):
                outputs.append(
                    builder.build(
                        0,
                        local_common,
                        pcp_context=context,
                        pcp_cache_group_idx=group,
                        num_actual_reqs=1,
                        common_ratio_to_sas_metadata=shared,
                        full_graph_mode=full_graph,
                    )
                )
                # MRV2 carries this attribute forward to the next cache group.
                self.assertIs(builder.common_ratio_to_sas_metadata, shared)
                shared = builder.common_ratio_to_sas_metadata
            runs.append((shared, outputs))
        return builders, runs

    def test_pcp_shares_global_results_and_isolates_local_layout(self):
        for prefill, sharded in ((True, False), (False, True)):
            with self.subTest(prefill=prefill, sharded=sharded):
                self.sas_op.reset_mock()
                self.qli_op.reset_mock()
                self.rope_op.reset_mock()
                builders, runs = self.run_pcp_groups(prefill=prefill, sharded=sharded)
                shared, outputs = runs[0]
                self.assertEqual(self.sas_op.call_count, 2)
                self.assertEqual(self.qli_op.call_count, 2)
                self.assertEqual(self.rope_op.call_count, 2)
                self.assertIsNot(shared["pcp_global"], shared["pcp_local"])
                for group, output in enumerate(outputs):
                    global_req = output.global_dsa_metadata.req_metadata
                    local_req = output.req_metadata
                    torch.testing.assert_close(global_req.seq_lens, torch.tensor([10], dtype=torch.int32))
                    torch.testing.assert_close(local_req.seq_lens, torch.tensor([6], dtype=torch.int32))
                    self.assertEqual(global_req.block_table[0, 0].item(), 10 + group)
                    self.assertEqual(local_req.block_table[0, 0].item(), 30 + group)
                    torch.testing.assert_close(global_req.qli_seqused_k, torch.tensor([2], dtype=torch.int32))
                    torch.testing.assert_close(local_req.qli_seqused_k, torch.tensor([1], dtype=torch.int32))
                    self.assertIs(
                        builders[group]._global_metadata_builder.common_ratio_to_sas_metadata, shared["pcp_global"]
                    )
                first, second = [output.global_dsa_metadata.req_metadata for output in outputs]
                self.assertNotEqual(first.sas_metadata.data_ptr(), second.sas_metadata.data_ptr())

    def test_pcp_recomputes_results_each_invocation(self):
        _, runs = self.run_pcp_groups(steps=2)
        self.assertEqual(self.sas_op.call_count, 4)
        self.assertEqual(self.qli_op.call_count, 4)
        self.assertEqual(self.rope_op.call_count, 4)
        self.assertIsNot(runs[0][0]["pcp_global"], runs[1][0]["pcp_global"])
        self.assertIsNot(runs[0][0]["pcp_local"], runs[1][0]["pcp_local"])

    def test_replicated_decode_uses_only_local_metadata(self):
        builders, runs = self.run_pcp_groups(prefill=False)
        shared, outputs = runs[0]
        self.assertEqual(shared["pcp_global"], {})
        self.assertTrue(all(type(output) is dsa_v1.AscendDSAMetadata for output in outputs))
        for builder in builders:
            builder._build_global_common_attn_metadata.assert_not_called()
        self.sas_op.assert_called_once()
        self.qli_op.assert_called_once()
        self.rope_op.assert_called_once()

    def test_empty_local_rank_still_shares_global_metadata(self):
        _, runs = self.run_pcp_groups(empty_local=True)
        shared, outputs = runs[0]
        self.assertEqual(shared["pcp_local"], {})
        self.assertTrue(all(output.req_metadata is None for output in outputs))
        self.sas_op.assert_called_once()
        self.qli_op.assert_called_once()

    def test_sharded_graph_keeps_per_builder_rope_buffers(self):
        proxy = Mock()
        proxy.copy_to_buffers.side_effect = lambda buffers, size: buffers.setdefault(
            "rope", (torch.ones(size), torch.zeros(size))
        )
        self.rope_op.return_value = (proxy, object())
        builders, runs = self.run_pcp_groups(prefill=False, sharded=True, full_graph=True)
        self.assertEqual(proxy.copy_to_buffers.call_count, 2)
        outputs = runs[0][1]
        for builder, output in zip(builders, outputs):
            self.assertIs(output.global_dsa_metadata.req_metadata.cos, builder._global_rope_buffers["rope"][0])
        self.assertNotEqual(
            outputs[0].global_dsa_metadata.req_metadata.cos.data_ptr(),
            outputs[1].global_dsa_metadata.req_metadata.cos.data_ptr(),
        )
