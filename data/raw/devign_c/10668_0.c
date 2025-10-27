static int decode_svq1_block (bit_buffer_t *bitbuf, uint8_t *pixels, int pitch, int intra) {

  uint32_t    bit_cache;

  vlc_code_t *vlc;

  uint8_t    *list[63];

  uint32_t   *dst;

  uint32_t   *codebook;

  int	      entries[6];

  int	      i, j, m, n;

  int	      mean, stages;

  int	      x, y, width, height, level;

  uint32_t    n1, n2, n3, n4;



  /* initialize list for breadth first processing of vectors */

  list[0] = pixels;



  /* recursively process vector */

  for (i=0, m=1, n=1, level=5; i < n; i++) {

    for (; level > 0; i++) {



      /* process next depth */

      if (i == m) {

	m = n;



	if (--level == 0)

	  break;

      }



      /* divide block if next bit set */

      if (get_bits (bitbuf, 1) == 0)

	break;



      /* add child nodes */

      list[n++] = list[i];

      list[n++] = list[i] + (((level & 1) ? pitch : 1) << ((level / 2) + 1));

    }



    /* destination address and vector size */

    dst = (uint32_t *) list[i];

    width = 1 << ((4 + level) /2);

    height = 1 << ((3 + level) /2);



    /* get number of stages (-1 skips vector, 0 for mean only) */

    bit_cache = get_bit_cache (bitbuf);



    if (intra)

      vlc = &intra_vector_tables[level][bit_cache >> (32 - 7)];

    else

      vlc = &inter_vector_tables[level][bit_cache >> (32 - 6)];



    /* flush bits */

    stages	    = vlc->value;

    skip_bits(bitbuf,vlc->length);



    if (stages == -1) {

      if (intra) {

	for (y=0; y < height; y++) {

	  memset (&dst[y*(pitch / 4)], 0, width);

	}

      }

      continue;		/* skip vector */

    }



    if ((stages > 0) && (level >= 4)) {

#ifdef DEBUG_SVQ1

    printf("Error (decode_svq1_block): invalid vector: stages=%i level=%i\n",stages,level);

#endif

      return -1;	/* invalid vector */

    }



    /* get mean value for vector */

    bit_cache = get_bit_cache (bitbuf);



    if (intra) {

      if (bit_cache >= 0x25000000)

	vlc = &intra_mean_table_0[(bit_cache >> (32 - 8)) - 37];

      else if (bit_cache >= 0x03400000)

	vlc = &intra_mean_table_1[(bit_cache >> (32 - 10)) - 13];

      else if (bit_cache >= 0x00040000) 

	vlc = &intra_mean_table_2[(bit_cache >> (32 - 14)) - 1];

      else

	vlc = &intra_mean_table_3[bit_cache >> (32 - 20)];

    } else {

      if (bit_cache >= 0x0B000000)

	vlc = &inter_mean_table_0[(bit_cache >> (32 - 8)) - 11];

      else if (bit_cache >= 0x01200000)

	vlc = &inter_mean_table_1[(bit_cache >> (32 - 12)) - 18];

      else if (bit_cache >= 0x002E0000) 

	vlc = &inter_mean_table_2[(bit_cache >> (32 - 15)) - 23];

      else if (bit_cache >= 0x00094000)

	vlc = &inter_mean_table_3[(bit_cache >> (32 - 18)) - 37];

      else if (bit_cache >= 0x00049000)

	vlc = &inter_mean_table_4[(bit_cache >> (32 - 20)) - 73];

      else

	vlc = &inter_mean_table_5[bit_cache >> (32 - 22)];

    }



    /* flush bits */

    mean	    = vlc->value;

    skip_bits(bitbuf,vlc->length);



    if (intra && stages == 0) {

      for (y=0; y < height; y++) {

	memset (&dst[y*(pitch / 4)], mean, width);

      }

    } else {

      codebook = (uint32_t *) (intra ? intra_codebooks[level] : inter_codebooks[level]);

      bit_cache = get_bits (bitbuf, 4*stages);



      /* calculate codebook entries for this vector */

      for (j=0; j < stages; j++) {

	entries[j] = (((bit_cache >> (4*(stages - j - 1))) & 0xF) + 16*j) << (level + 1);

      }



      mean -= (stages * 128);

      n4    = ((mean + (mean >> 31)) << 16) | (mean & 0xFFFF);



      for (y=0; y < height; y++) {

	for (x=0; x < (width / 4); x++, codebook++) {

	  if (intra) {

	    n1 = n4;

	    n2 = n4;

	  } else {

	    n3 = dst[x];



	    /* add mean value to vector */

	    n1 = ((n3 & 0xFF00FF00) >> 8) + n4;

	    n2 =  (n3 & 0x00FF00FF)	  + n4;

	  }



	  /* add codebook entries to vector */

	  for (j=0; j < stages; j++) {

	    n3  = codebook[entries[j]] ^ 0x80808080;

	    n1 += ((n3 & 0xFF00FF00) >> 8);

	    n2 +=  (n3 & 0x00FF00FF);

	  }



	  /* clip to [0..255] */

	  if (n1 & 0xFF00FF00) {

	    n3  = ((( n1 >> 15) & 0x00010001) | 0x01000100) - 0x00010001;

	    n1 += 0x7F007F00;

	    n1 |= (((~n1 >> 15) & 0x00010001) | 0x01000100) - 0x00010001;

	    n1 &= (n3 & 0x00FF00FF);

	  }



	  if (n2 & 0xFF00FF00) {

	    n3  = ((( n2 >> 15) & 0x00010001) | 0x01000100) - 0x00010001;

	    n2 += 0x7F007F00;

	    n2 |= (((~n2 >> 15) & 0x00010001) | 0x01000100) - 0x00010001;

	    n2 &= (n3 & 0x00FF00FF);

	  }



	  /* store result */

	  dst[x] = (n1 << 8) | n2;

        }



        dst += (pitch / 4);

      }

    }

  }



  return 0;

}
