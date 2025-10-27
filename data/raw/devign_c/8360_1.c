av_cold int ff_rdft_init(RDFTContext *s, int nbits, enum RDFTransformType trans)

{

    int n = 1 << nbits;

    int i;

    const double theta = (trans == RDFT || trans == IRIDFT ? -1 : 1)*2*M_PI/n;



    s->nbits           = nbits;

    s->inverse         = trans == IRDFT || trans == IRIDFT;

    s->sign_convention = trans == RIDFT || trans == IRIDFT ? 1 : -1;



    if (nbits < 4 || nbits > 16)

        return -1;



    if (ff_fft_init(&s->fft, nbits-1, trans == IRDFT || trans == RIDFT) < 0)

        return -1;



    s->tcos = ff_cos_tabs[nbits-4];

    s->tsin = ff_sin_tabs[nbits-4]+(trans == RDFT || trans == IRIDFT)*(n>>2);

    for (i = 0; i < (n>>2); i++) {

        s->tcos[i] = cos(i*theta);

        s->tsin[i] = sin(i*theta);

    }

    return 0;

}
