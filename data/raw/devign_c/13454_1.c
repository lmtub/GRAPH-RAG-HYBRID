AVFixedDSPContext * avpriv_alloc_fixed_dsp(int bit_exact)
{
    AVFixedDSPContext * fdsp = av_malloc(sizeof(AVFixedDSPContext));
    fdsp->vector_fmul_window_scaled = vector_fmul_window_fixed_scaled_c;
    fdsp->vector_fmul_window = vector_fmul_window_fixed_c;
    return fdsp;
}