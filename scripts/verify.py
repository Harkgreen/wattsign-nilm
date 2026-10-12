"""Explicit non-fitting checks. Synthetic optimizer tests require separate opt-in."""
from pathlib import Path
import argparse
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NPU_CHECKS = {'test_core.Tests': ['test_adc',
                     'test_nonactive_and_phase',
                     'test_trajectory',
                     'test_baseline_no_future',
                     'test_token',
                     'test_individual_label_boundaries',
                     'test_threshold',
                     'test_weighted_bootstrap',
                     'test_power_bins',
                     'test_models_and_initialization',
                     'test_causal_backbones',
                     'test_pairing_and_missing',
                     'test_block_bootstrap_repeatable'],
 'test_gru.GRUTests': ['test_forward_hidden_input_and_parameter_gradients',
                       'test_eval_dropout_checkpoint_and_no_fused_dispatch',
                       'test_model_parameter_and_mac_count_unchanged',
                       'test_reject_half_precision'],
 'test_label_upgrade.LabelUpgradeTests': ['test_invalid_interval_and_both_boundary_masks',
                                          'test_leading_trailing_consecutive_invalid_and_unsupported',
                                          'test_legal_single_and_aggregate_keep_original_interface',
                                          'test_skip_preserves_actual_receipt_without_calling_device'],
 'test_launcher.LauncherTests': ['test_help_and_invalid_arguments_do_not_run_python',
                                 'test_cli_overrides_environment_for_prepare_and_run',
                                 'test_defaults_environment_and_equals_syntax',
                                 'test_relative_path_resolved_before_changing_directory',
                                 'test_outside_tmux_launches_one_session_without_running_python',
                                 'test_existing_session_does_not_restart_pipeline',
                                 'test_memory_options_forwarded_into_tmux',
                                 'test_skip_preflight_is_explicit_and_forwarded_to_tmux',
                                 'test_stale_session_stops_before_pipeline',
                                 'test_missing_tmux_stops_before_pipeline'],
 'test_launcher.SessionGuardTests': ['test_requires_live_session',
                                     'test_training_commands_block_before_backend_import'],
 'test_memory.MemoryTests': ['test_ram_readonly_reuse_budget_and_low_memory',
                             'test_prefetch_exception_is_not_hidden',
                             'test_resident_window_gather_matches_reference',
                             'test_invalid_limits_fail',
                             'test_device_cache_respects_budget_and_free_memory'],
 'test_preflight.PreflightTests': ['test_subprocess_failure_timeout_stale_receipt_and_continuation',
                                   'test_dependency_check_reports_missing_reportlab',
                                   'test_failed_report_preserves_primary_error_and_pack_contains_diagnostics'],
 'test_local_statistics.FrozenAnalysisContracts': ['test_paired_ties_and_unknown_masks',
                                                   'test_invalid_draws_keep_identical_rng_order',
                                                   'test_cumsum_fallback_preserves_weighted_ties',
                                                   'test_absent_uses_all_eligible_heads_and_per_class_intervals',
                                                   'test_absent_invalid_samples_keep_original_attempts',
                                                   'test_absent_head_without_known_off_support_is_not_silently_dropped',
                                                   'test_training_or_npu_completion_claims_are_rejected_without_writes',
                                                   'test_original_protocol_constants_remain_frozen',
                                                   'test_adapter_does_not_import_training_or_npu_or_call_threshold_selection'],
 'test_npu_portability.NpuPortabilityTests': ['test_inputs_and_outputs_cannot_overlap',
                                              'test_task_path_rejects_escape',
                                              'test_unreviewed_pickle_is_rejected_before_loading',
                                              'test_cpu_storage_bounds_and_hooks_are_checked',
                                              'test_readout_plan_is_really_read_only',
                                              'test_converter_missing_input_never_creates_output',
                                              'test_cli_help_works_outside_repository',
                                              'test_inference_requires_explicit_cpu_or_cuda_device',
                                              'test_compact_metadata_preserves_masks_and_integer_endpoints',
                                              'test_compact_metadata_rejects_duplicate_endpoints',
                                              'test_compact_manifest_rejects_changed_file_without_writes']}

# Purely mocked runtime/report checks; these never launch training or analysis.
NPU_CHECKS['test_optional_reporting.OptionalReportingTests'] = [
    'test_no_font_run_packs_completed_results_without_rendering',
    'test_requested_font_validates_before_queue_then_renders_and_packs',
    'test_missing_font_and_rendering_dependency_stop_without_output_writes',
    'test_direct_report_without_font_creates_no_output_directory',
    'test_render_failure_still_packs_without_marking_completed_training_blocked',
    'test_existing_decision_is_reused_and_partial_queue_does_not_analyze',
    'test_nonrendered_pack_contains_saved_metrics_without_weights_or_predictions',
    'test_analysis_failure_still_packs_with_correct_error_stage',
    'test_report_and_pack_failures_keep_original_error_and_separate_pack_diagnostic',
    'test_pack_only_failure_is_recorded_as_pack',
]
NPU_CHECKS['test_launcher.LauncherTests'] += [
    'test_font_is_optional_and_forwarded_with_cli_precedence',
    'test_failed_queue_packs_without_font_and_forwards_requested_report_font',
]

NO_FIT_CHECKS = {
    "test_mvp_preprocessing.MaskedStandardizerTests.test_fit_uses_only_observed_training_values",
    "test_mvp_preprocessing.MaskedStandardizerTests.test_missing_zero_and_observed_standardised_zero_remain_distinguishable",
    "test_mvp_preprocessing.MaskedStandardizerTests.test_serialisation_round_trip_preserves_transform",
    "test_mvp_preprocessing.MaskedStandardizerTests.test_observed_nonfinite_value_is_rejected",
    "test_online_training.OnlineFeatureTableLoadingTests.test_scaler_uses_unweighted_training_observations_only",
}

def _skip_fit_checks(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            _skip_fit_checks(test)
        elif test.id() in NO_FIT_CHECKS:
            method = getattr(type(test), test._testMethodName)
            method.__unittest_skip__ = True
            method.__unittest_skip_why__ = "Default verification does not call fit, including synthetic scaler statistics"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npu',action='store_true',help='CPU math, memory and mocked launcher checks; no hardware or fitting')
    args=parser.parse_args()
    os.environ.pop('LSA_RUN_TRAINING_TESTS',None)
    sys.dont_write_bytecode=True
    if args.npu:
        root=ROOT/'npu/tdha_multilabel_v3'
        sys.path[:0]=[str(root),str(root/'tests'),str(ROOT/'tests')]
        names=[prefix+'.'+method for prefix,methods in NPU_CHECKS.items() for method in methods]
        suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    else:
        sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
        names=['test_'+name for name in ('event_detection','event_features','features','grouped_mlp','inventory','measurement_inventory','mvp_preprocessing','online_candidate_detection','online_features','online_samples','online_training','online_v11','refit_adapter','ukdale','ukdale_phase2','streaming_numerics','public_entrypoints','notebook_display_safety','notebook_helper_contracts','selected_models')]
        suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    _skip_fit_checks(suite)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':
    raise SystemExit(main())
