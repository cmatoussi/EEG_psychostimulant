### sensor_epoch
| Condition | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|---|---|---|---|---|---|
| EC_baseline | Cz — complexity_petrosian_fd: 3.608 | P4 — band_abs_beta: -3.462 | Cz — complexity_perm_entropy: -3.354 | C4 — band_abs_gamma: -3.123 | P4 — band_corr_abs_theta: 2.745 |
| EO_baseline | T5 — band_log_abs_theta: 1.667 | F8 — complexity_petrosian_fd: 1.575 | Fz — band_abs_delta: 1.548 | Pz — band_rel_beta: -1.469 | P4 — complexity_hjorth_mobility: 1.392 |
| HV_EC | Fp1 — band_abs_alpha: 2.015 | F4 — complexity_higuchi_fd: 1.817 | P3 — band_abs_alpha: -1.773 | T4 — band_log_abs_theta: -1.731 | C4 — band_corr_abs_theta: 1.714 |
| PHOTO_EC | F3 — complexity_petrosian_fd: 1.263 | F7 — band_corr_abs_theta: -1.158 | T5 — param_peak_freq_dom: 1.155 | C4 — complexity_perm_entropy: -1.099 | C4 — complexity_svd_entropy: -0.972 |
| PostHV_EC | F4 — complexity_petrosian_fd: -1.977 | F3 — complexity_higuchi_fd: 1.422 | F7 — band_abs_theta: -1.413 | Fz — band_log_abs_beta: -1.392 | T5 — band_corr_rel_alpha: 1.380 |
| PostHV_EO | F3 — complexity_hjorth_complexity: 0.944 | Fp1 — band_abs_theta: -0.720 | C4 — band_corr_log_abs_delta: 0.686 | T4 — band_rel_alpha: 0.682 | T3 — complexity_hurst_exponent: -0.661 |
| HV_EO | T6 — band_log_abs_gamma: 1.206 | F7 — complexity_perm_entropy: -1.110 | Fz — band_log_abs_gamma: 0.967 | T6 — complexity_higuchi_fd: 0.803 | Pz — band_rel_alpha: 0.740 |
| PHOTO_EO | T4 — complexity_hjorth_complexity: 1.338 | C3 — band_rel_alpha: -1.182 | P4 — band_log_abs_gamma: -1.142 | P3 — complexity_perm_entropy: 1.131 | Pz — param_peak_power_dom: -1.129 |

### sensor_subject
| Condition | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|---|---|---|---|---|---|
| EC_baseline | C3 — mad_complexity_hurst_exponent: -0.680 | Pz — median_band_corr_log_abs_gamma: -0.493 | C3 — mean_band_corr_log_abs_beta: 0.460 | T4 — iqr_param_peak_count: -0.446 | Fp1 — agg_band_corr_ratio_alpha_gamma: -0.432 |
| EO_baseline | O1 — iqr_band_corr_rel_delta: 0.643 | F4 — iqr_band_log_abs_theta: -0.630 | F7 — iqr_param_alpha_peak_freq: 0.488 | C3 — iqr_band_corr_log_abs_alpha: 0.481 | T5 — iqr_band_rel_alpha: -0.389 |
| HV_EC | P4 — median_param_peak_count: 0.452 | T3 — mad_complexity_hjorth_complexity: 0.411 | C3 — mean_band_log_abs_beta: 0.389 | F7 — iqr_band_corr_log_abs_gamma: -0.322 | T6 — median_complexity_kurtosis: -0.309 |
| PHOTO_EC | F3 — iqr_band_rel_theta: 0.645 | F3 — mad_complexity_higuchi_fd: 0.597 | F7 — median_param_peak_count: -0.506 | Fp2 — mean_band_corr_log_abs_beta: 0.416 | Fp1 — median_band_corr_rel_delta: -0.406 |
| PostHV_EC | P4 — mad_complexity_hurst_exponent: 0.543 | Fp2 — iqr_band_log_abs_alpha: 0.503 | O2 — iqr_band_corr_rel_alpha: -0.483 | P4 — iqr_param_peak_count: 0.460 | Fp2 — mad_complexity_spectral_entropy: -0.450 |
| PostHV_EO | T4 — mad_complexity_petrosian_fd: 0.500 | O1 — mean_param_peak_bandwidth_dom: 0.480 | C4 — iqr_band_log_abs_theta: -0.379 | Fz — median_param_peak_bandwidth_dom: 0.343 | C4 — median_param_alpha_peak_freq: -0.311 |
| HV_EO | O2 — mad_complexity_spectral_entropy: -0.564 | Pz — mean_param_fit_error: 0.553 | Fp2 — iqr_band_corr_log_abs_beta: -0.274 | O2 — median_band_corr_log_abs_gamma: 0.271 | F3 — mean_param_peak_count: -0.228 |
| PHOTO_EO | O1 — mad_complexity_dispersion_entropy: 0.763 | Fz — median_param_peak_bandwidth_dom: 0.648 | T4 — mad_complexity_kurtosis: -0.612 | Fp1 — iqr_param_alpha_peak_power: -0.474 | O2 — iqr_param_peak_power_dom: -0.394 |

### pooled_epoch
| Condition | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|---|---|---|---|---|---|
| EC_baseline | complexity_petrosian_fd_chgrp-central_midline: 2.873 | complexity_perm_entropy_chgrp-central_midline: -2.850 | band_abs_delta_chgrp-front_midline: -2.436 | complexity_petrosian_fd_chgrp-central_right: -2.008 | band_log_abs_alpha_chgrp-posterior_right: -1.811 |
| EO_baseline | band_corr_abs_beta_chgrp-front_left: 2.611 | complexity_perm_entropy_chgrp-central_midline: -2.079 | param_offset_chgrp-posterior_midline: -2.027 | band_log_abs_beta_chgrp-posterior_midline: 2.025 | complexity_petrosian_fd_chgrp-central_midline: 1.930 |
| HV_EC | band_log_abs_delta_chgrp-front_midline: 2.019 | complexity_petrosian_fd_chgrp-front_midline: -1.708 | band_log_abs_delta_chgrp-front_right: -1.675 | band_abs_beta_chgrp-front_left: 1.565 | band_log_abs_delta_chgrp-posterior_right: -1.556 |
| PHOTO_EC | param_peak_power_dom_chgrp-posterior_left: -1.328 | band_abs_alpha_chgrp-posterior_left: -1.316 | band_log_abs_gamma_chgrp-posterior_right: -1.216 | band_corr_abs_gamma_chgrp-front_right: -1.001 | band_rel_alpha_chgrp-posterior_left: 0.965 |
| PostHV_EC | band_corr_log_abs_alpha_chgrp-posterior_left: -1.709 | complexity_petrosian_fd_chgrp-front_right: -1.574 | band_log_abs_alpha_chgrp-central_right: 1.378 | band_log_abs_beta_chgrp-front_midline: -1.342 | band_abs_theta_chgrp-front_left: -1.334 |
| PostHV_EO | band_log_abs_beta_chgrp-posterior_midline: 1.195 | band_log_abs_theta_chgrp-posterior_midline: -1.141 | band_rel_beta_chgrp-posterior_midline: -1.007 | complexity_petrosian_fd_chgrp-central_left: 0.964 | band_corr_rel_theta_chgrp-posterior_midline: 0.963 |
| HV_EO | param_r_squared_chgrp-central_left: 1.315 | complexity_perm_entropy_chgrp-front_left: -1.219 | complexity_perm_entropy_chgrp-central_midline: 1.212 | param_exponent_chgrp-front_midline: -1.209 | band_rel_alpha_chgrp-posterior_midline: 1.191 |
| PHOTO_EO | band_log_abs_theta_chgrp-posterior_left: 1.497 | complexity_hjorth_complexity_chgrp-central_right: 1.369 | complexity_perm_entropy_chgrp-posterior_left: 1.282 | band_rel_alpha_chgrp-central_left: -1.200 | complexity_zero_crossings_chgrp-posterior_right: -1.069 |

### pooled_subject
| Condition | Feature 1 | Feature 2 | Feature 3 | Feature 4 | Feature 5 |
|---|---|---|---|---|---|
| EC_baseline | iqr_band_rel_alpha_chgrp-front_left: -0.713 | mad_complexity_spectral_entropy_chgrp-posterior_midline: 0.536 | median_band_corr_log_abs_gamma_chgrp-posterior_midline: -0.528 | iqr_band_corr_log_abs_gamma_chgrp-front_right: -0.524 | iqr_band_rel_beta_chgrp-front_midline: -0.486 |
| EO_baseline | iqr_param_alpha_peak_freq_chgrp-central_left: 0.585 | agg_band_corr_ratio_beta_gamma_chgrp-posterior_left: 0.565 | iqr_param_peak_count_chgrp-central_midline: -0.539 | median_complexity_spectral_entropy_chgrp-posterior_right: -0.528 | iqr_param_exponent_chgrp-posterior_midline: -0.469 |
| HV_EC | mean_param_peak_count_chgrp-front_midline: 0.682 | mean_complexity_kurtosis_chgrp-central_left: -0.571 | iqr_param_peak_count_chgrp-front_midline: 0.557 | mad_complexity_hurst_exponent_chgrp-front_right: 0.501 | iqr_param_peak_count_chgrp-central_left: 0.409 |
| PHOTO_EC | mad_complexity_perm_entropy_chgrp-central_midline: 0.878 | iqr_param_peak_count_chgrp-posterior_left: -0.539 | iqr_band_corr_rel_beta_chgrp-central_right: -0.517 | iqr_band_log_abs_beta_chgrp-posterior_midline: -0.454 | mad_complexity_sample_entropy_chgrp-front_left: 0.452 |
| PostHV_EC | iqr_band_corr_rel_alpha_chgrp-posterior_right: -0.532 | iqr_param_peak_bandwidth_dom_chgrp-posterior_left: -0.524 | mean_band_corr_log_abs_gamma_chgrp-posterior_left: 0.517 | iqr_band_corr_log_abs_theta_chgrp-posterior_left: 0.515 | iqr_param_fit_error_chgrp-posterior_right: 0.448 |
| PostHV_EO | iqr_band_log_abs_theta_chgrp-central_right: -0.644 | mad_complexity_hurst_exponent_chgrp-posterior_midline: 0.625 | iqr_param_peak_count_chgrp-posterior_midline: 0.553 | iqr_band_rel_beta_chgrp-central_midline: -0.455 | median_band_corr_log_abs_gamma_chgrp-front_right: 0.379 |
| HV_EO | mean_param_fit_error_chgrp-posterior_midline: 0.717 | mad_complexity_hjorth_complexity_chgrp-posterior_left: -0.540 | mean_band_corr_log_abs_delta_chgrp-posterior_left: 0.529 | mean_param_peak_bandwidth_dom_chgrp-central_left: 0.394 | mean_param_peak_count_chgrp-front_left: -0.322 |
| PHOTO_EO | median_param_peak_bandwidth_dom_chgrp-front_midline: 0.737 | mad_complexity_svd_entropy_chgrp-front_right: -0.686 | iqr_band_rel_beta_chgrp-front_left: -0.666 | iqr_band_rel_delta_chgrp-front_midline: 0.595 | iqr_param_peak_bandwidth_dom_chgrp-front_right: 0.523 |

