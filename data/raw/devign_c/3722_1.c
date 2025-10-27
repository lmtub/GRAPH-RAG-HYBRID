static void decode_mclms(WmallDecodeCtx *s)

{

    s->mclms_order = (get_bits(&s->gb, 4) + 1) * 2;

    s->mclms_scaling = get_bits(&s->gb, 4);

    if(get_bits1(&s->gb)) {

	// mclms_send_coef

	int i;

	int send_coef_bits;

	int cbits = av_log2(s->mclms_scaling + 1);

	assert(cbits == my_log2(s->mclms_scaling + 1));

	if(1 << cbits < s->mclms_scaling + 1)

	    cbits++;



	send_coef_bits = (cbits ? get_bits(&s->gb, cbits) : 0) + 2;



	for(i = 0; i < s->mclms_order * s->num_channels * s->num_channels; i++) {

	    s->mclms_coeffs[i] = get_bits(&s->gb, send_coef_bits);

	}



	for(i = 0; i < s->num_channels; i++) {

	    int c;

	    for(c = 0; c < i; c++) {

		s->mclms_coeffs_cur[i * s->num_channels + c] = get_bits(&s->gb, send_coef_bits);

	    }

	}

    }

}
