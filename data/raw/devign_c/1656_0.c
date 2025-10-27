static void dv_decode_ac(GetBitContext *gb, BlockInfo *mb, DCTELEM *block)

{

    int last_index = get_bits_size(gb);

    const uint8_t *scan_table = mb->scan_table;

    const uint8_t *shift_table = mb->shift_table;

    int pos = mb->pos;

    int partial_bit_count = mb->partial_bit_count;

    int level, pos1, run, vlc_len, index;

    

    OPEN_READER(re, gb);

    UPDATE_CACHE(re, gb);

    

    /* if we must parse a partial vlc, we do it here */

    if (partial_bit_count > 0) {

        re_cache = ((unsigned)re_cache >> partial_bit_count) |

	           (mb->partial_bit_buffer << (sizeof(re_cache)*8 - partial_bit_count));

	re_index -= partial_bit_count;

	mb->partial_bit_count = 0;

    }



    /* get the AC coefficients until last_index is reached */

    for(;;) {

#ifdef VLC_DEBUG

        printf("%2d: bits=%04x index=%d\n", pos, SHOW_UBITS(re, gb, 16), re_index);

#endif

        /* our own optimized GET_RL_VLC */

        index = NEG_USR32(re_cache, TEX_VLC_BITS);

	vlc_len = dv_rl_vlc[index].len;

        if (vlc_len < 0) {

            index = NEG_USR32((unsigned)re_cache << TEX_VLC_BITS, -vlc_len) + dv_rl_vlc[index].level;

            vlc_len = TEX_VLC_BITS - vlc_len;

        }

        level = dv_rl_vlc[index].level;

	run = dv_rl_vlc[index].run;

	

	/* gotta check if we're still within gb boundaries */

	if (re_index + vlc_len > last_index) {

	    /* should be < 16 bits otherwise a codeword could have been parsed */

	    mb->partial_bit_count = last_index - re_index;

	    mb->partial_bit_buffer = NEG_USR32(re_cache, mb->partial_bit_count);

	    re_index = last_index;

	    break;

	}

	re_index += vlc_len;



#ifdef VLC_DEBUG

	printf("run=%d level=%d\n", run, level);

#endif

	pos += run; 	

	if (pos >= 64)

	    break;

        

	if (level) {

            pos1 = scan_table[pos];

            block[pos1] = level << shift_table[pos1];

        } 



        UPDATE_CACHE(re, gb);

    }

    CLOSE_READER(re, gb);

    mb->pos = pos;

}
