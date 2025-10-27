static int vga_osi_call (CPUState *env)

{

    static int vga_vbl_enabled;

    int linesize;



#if 0

    printf("osi_call R5=%016" PRIx64 "\n", ppc_dump_gpr(env, 5));

#endif



    /* same handler as PearPC, coming from the original MOL video

       driver. */

    switch(env->gpr[5]) {

    case 4:

        break;

    case 28: /* set_vmode */

        if (env->gpr[6] != 1 || env->gpr[7] != 0)

            env->gpr[3] = 1;

        else

            env->gpr[3] = 0;

        break;

    case 29: /* get_vmode_info */

        if (env->gpr[6] != 0) {

            if (env->gpr[6] != 1 || env->gpr[7] != 0) {

                env->gpr[3] = 1;

                break;

            }

        }

        env->gpr[3] = 0;

        env->gpr[4] = (1 << 16) | 1; /* num_vmodes, cur_vmode */

        env->gpr[5] = (1 << 16) | 0; /* num_depths, cur_depth_mode */

        env->gpr[6] = (graphic_width << 16) | graphic_height; /* w, h */

        env->gpr[7] = 85 << 16; /* refresh rate */

        env->gpr[8] = (graphic_depth + 7) & ~7; /* depth (round to byte) */

        linesize = ((graphic_depth + 7) >> 3) * graphic_width;

        linesize = (linesize + 3) & ~3;

        env->gpr[9] = (linesize << 16) | 0; /* row_bytes, offset */

        break;

    case 31: /* set_video power */

        env->gpr[3] = 0;

        break;

    case 39: /* video_ctrl */

        if (env->gpr[6] == 0 || env->gpr[6] == 1)

            vga_vbl_enabled = env->gpr[6];

        env->gpr[3] = 0;

        break;

    case 47:

        break;

    case 59: /* set_color */

        /* R6 = index, R7 = RGB */

        env->gpr[3] = 0;

        break;

    case 64: /* get color */

        /* R6 = index */

        env->gpr[3] = 0;

        break;

    case 116: /* set hwcursor */

        /* R6 = x, R7 = y, R8 = visible, R9 = data */

        break;

    default:

        fprintf(stderr, "unsupported OSI call R5=%016" PRIx64 "\n",

                ppc_dump_gpr(env, 5));

        break;

    }



    return 1; /* osi_call handled */

}
