av_cold int ff_dcaadpcm_init(DCAADPCMEncContext *s)
{
    if (!s)
        return -1;
    s->private_data = av_malloc(sizeof(premultiplied_coeffs) * DCA_ADPCM_VQCODEBOOK_SZ);
    precalc(s->private_data);
    return 0;
}