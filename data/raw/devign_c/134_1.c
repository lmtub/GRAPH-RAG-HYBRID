int yuv2rgb_c_init_tables (SwsContext *c, const int inv_table[4], int fullRange, int brightness, int contrast, int saturation)

{

    const int isRgb = isBGR(c->dstFormat);

    const int bpp = fmt_depth(c->dstFormat);

    int i;

    uint8_t table_Y[1024];

    uint32_t *table_32 = 0;

    uint16_t *table_16 = 0;

    uint8_t *table_8 = 0;

    uint8_t *table_332 = 0;

    uint8_t *table_121 = 0;

    uint8_t *table_1 = 0;

    int entry_size = 0;

    void *table_r = 0, *table_g = 0, *table_b = 0;

    void *table_start;



    int64_t crv =  inv_table[0];

    int64_t cbu =  inv_table[1];

    int64_t cgu = -inv_table[2];

    int64_t cgv = -inv_table[3];

    int64_t cy  = 1<<16;

    int64_t oy  = 0;



//printf("%lld %lld %lld %lld %lld\n", cy, crv, cbu, cgu, cgv);

    if(!fullRange){

	cy= (cy*255) / 219;

	oy= 16<<16;

    }else{

        crv= (crv*224) / 255;

        cbu= (cbu*224) / 255;

        cgu= (cgu*224) / 255;

        cgv= (cgv*224) / 255;

    }



    cy = (cy *contrast             )>>16;

    crv= (crv*contrast * saturation)>>32;

    cbu= (cbu*contrast * saturation)>>32;

    cgu= (cgu*contrast * saturation)>>32;

    cgv= (cgv*contrast * saturation)>>32;

//printf("%lld %lld %lld %lld %lld\n", cy, crv, cbu, cgu, cgv);

    oy -= 256*brightness;



    for (i = 0; i < 1024; i++) {

	int j;



	j= (cy*(((i - 384)<<16) - oy) + (1<<31))>>32;

	j = (j < 0) ? 0 : ((j > 255) ? 255 : j);

	table_Y[i] = j;

    }



    switch (bpp) {

    case 32:

	table_start= table_32 = av_malloc ((197 + 2*682 + 256 + 132) * sizeof (uint32_t));



	entry_size = sizeof (uint32_t);

	table_r = table_32 + 197;

	table_b = table_32 + 197 + 685;

	table_g = table_32 + 197 + 2*682;



	for (i = -197; i < 256+197; i++)

	    ((uint32_t *)table_r)[i] = table_Y[i+384] << (isRgb ? 16 : 0);

	for (i = -132; i < 256+132; i++)

	    ((uint32_t *)table_g)[i] = table_Y[i+384] << 8;

	for (i = -232; i < 256+232; i++)

	    ((uint32_t *)table_b)[i] = table_Y[i+384] << (isRgb ? 0 : 16);

	break;



    case 24:

	table_start= table_8 = av_malloc ((256 + 2*232) * sizeof (uint8_t));



	entry_size = sizeof (uint8_t);

	table_r = table_g = table_b = table_8 + 232;



	for (i = -232; i < 256+232; i++)

	    ((uint8_t * )table_b)[i] = table_Y[i+384];

	break;



    case 15:

    case 16:

	table_start= table_16 = av_malloc ((197 + 2*682 + 256 + 132) * sizeof (uint16_t));



	entry_size = sizeof (uint16_t);

	table_r = table_16 + 197;

	table_b = table_16 + 197 + 685;

	table_g = table_16 + 197 + 2*682;



	for (i = -197; i < 256+197; i++) {

	    int j = table_Y[i+384] >> 3;



	    if (isRgb)

		j <<= ((bpp==16) ? 11 : 10);



	    ((uint16_t *)table_r)[i] = j;

	}

	for (i = -132; i < 256+132; i++) {

	    int j = table_Y[i+384] >> ((bpp==16) ? 2 : 3);



	    ((uint16_t *)table_g)[i] = j << 5;

	}

	for (i = -232; i < 256+232; i++) {

	    int j = table_Y[i+384] >> 3;



	    if (!isRgb)

		j <<= ((bpp==16) ? 11 : 10);



	    ((uint16_t *)table_b)[i] = j;

	}

	break;



    case 8:

	table_start= table_332 = av_malloc ((197 + 2*682 + 256 + 132) * sizeof (uint8_t));



	entry_size = sizeof (uint8_t);

	table_r = table_332 + 197;

	table_b = table_332 + 197 + 685;

	table_g = table_332 + 197 + 2*682;



	for (i = -197; i < 256+197; i++) {

	    int j = (table_Y[i+384 - 16] + 18)/36;



	    if (isRgb)

		j <<= 5;



	    ((uint8_t *)table_r)[i] = j;

	}

	for (i = -132; i < 256+132; i++) {

	    int j = (table_Y[i+384 - 16] + 18)/36;



	    if (!isRgb)

		j <<= 1;



	    ((uint8_t *)table_g)[i] = j << 2;

	}

	for (i = -232; i < 256+232; i++) {

	    int j = (table_Y[i+384 - 37] + 43)/85;



	    if (!isRgb)

		j <<= 6;



	    ((uint8_t *)table_b)[i] = j;

	}

	break;

    case 4:

    case 4|128:

	table_start= table_121 = av_malloc ((197 + 2*682 + 256 + 132) * sizeof (uint8_t));



	entry_size = sizeof (uint8_t);

	table_r = table_121 + 197;

	table_b = table_121 + 197 + 685;

	table_g = table_121 + 197 + 2*682;



	for (i = -197; i < 256+197; i++) {

	    int j = table_Y[i+384 - 110] >> 7;



	    if (isRgb)

		j <<= 3;



	    ((uint8_t *)table_r)[i] = j;

	}

	for (i = -132; i < 256+132; i++) {

	    int j = (table_Y[i+384 - 37]+ 43)/85;



	    ((uint8_t *)table_g)[i] = j << 1;

	}

	for (i = -232; i < 256+232; i++) {

	    int j =table_Y[i+384 - 110] >> 7;



	    if (!isRgb)

		j <<= 3;



	    ((uint8_t *)table_b)[i] = j;

	}

	break;



    case 1:

	table_start= table_1 = av_malloc (256*2 * sizeof (uint8_t));



	entry_size = sizeof (uint8_t);

	table_g = table_1;

	table_r = table_b = NULL;



	for (i = 0; i < 256+256; i++) {

	    int j = table_Y[i + 384 - 110]>>7;



	    ((uint8_t *)table_g)[i] = j;

	}

	break;



    default:

	table_start= NULL;

	av_log(c, AV_LOG_ERROR, "%ibpp not supported by yuv2rgb\n", bpp);

	//free mem?

	return -1;

    }



    for (i = 0; i < 256; i++) {

	c->table_rV[i] = (uint8_t *)table_r + entry_size * div_round (crv * (i-128), 76309);

	c->table_gU[i] = (uint8_t *)table_g + entry_size * div_round (cgu * (i-128), 76309);

	c->table_gV[i] = entry_size * div_round (cgv * (i-128), 76309);

	c->table_bU[i] = (uint8_t *)table_b + entry_size * div_round (cbu * (i-128), 76309);

    }



    av_free(c->yuvTable);

    c->yuvTable= table_start;

    return 0;

}
