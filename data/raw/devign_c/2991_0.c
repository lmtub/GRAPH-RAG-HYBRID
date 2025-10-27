static AddressParts gen_lea_modrm_0(CPUX86State *env, DisasContext *s,

                                    int modrm)

{

    int def_seg, base, index, scale, mod, rm;

    target_long disp;

    bool havesib;



    def_seg = R_DS;

    index = -1;

    scale = 0;

    disp = 0;



    mod = (modrm >> 6) & 3;

    rm = modrm & 7;

    base = rm | REX_B(s);



    if (mod == 3) {

        /* Normally filtered out earlier, but including this path

           simplifies multi-byte nop, as well as bndcl, bndcu, bndcn.  */

        goto done;

    }



    switch (s->aflag) {

    case MO_64:

    case MO_32:

        havesib = 0;

        if (rm == 4) {

            int code = cpu_ldub_code(env, s->pc++);

            scale = (code >> 6) & 3;

            index = ((code >> 3) & 7) | REX_X(s);

            if (index == 4) {

                index = -1;  /* no index */

            }

            base = (code & 7) | REX_B(s);

            havesib = 1;

        }



        switch (mod) {

        case 0:

            if ((base & 7) == 5) {

                base = -1;

                disp = (int32_t)cpu_ldl_code(env, s->pc);

                s->pc += 4;

                if (CODE64(s) && !havesib) {

                    base = -2;

                    disp += s->pc + s->rip_offset;

                }

            }

            break;

        case 1:

            disp = (int8_t)cpu_ldub_code(env, s->pc++);

            break;

        default:

        case 2:

            disp = (int32_t)cpu_ldl_code(env, s->pc);

            s->pc += 4;

            break;

        }



        /* For correct popl handling with esp.  */

        if (base == R_ESP && s->popl_esp_hack) {

            disp += s->popl_esp_hack;

        }

        if (base == R_EBP || base == R_ESP) {

            def_seg = R_SS;

        }

        break;



    case MO_16:

        if (mod == 0) {

            if (rm == 6) {

                base = -1;

                disp = cpu_lduw_code(env, s->pc);

                s->pc += 2;

                break;

            }

        } else if (mod == 1) {

            disp = (int8_t)cpu_ldub_code(env, s->pc++);

        } else {

            disp = (int16_t)cpu_lduw_code(env, s->pc);

            s->pc += 2;

        }



        switch (rm) {

        case 0:

            base = R_EBX;

            index = R_ESI;

            break;

        case 1:

            base = R_EBX;

            index = R_EDI;

            break;

        case 2:

            base = R_EBP;

            index = R_ESI;

            def_seg = R_SS;

            break;

        case 3:

            base = R_EBP;

            index = R_EDI;

            def_seg = R_SS;

            break;

        case 4:

            base = R_ESI;

            break;

        case 5:

            base = R_EDI;

            break;

        case 6:

            base = R_EBP;

            def_seg = R_SS;

            break;

        default:

        case 7:

            base = R_EBX;

            break;

        }

        break;



    default:

        tcg_abort();

    }



 done:

    return (AddressParts){ def_seg, base, index, scale, disp };

}
