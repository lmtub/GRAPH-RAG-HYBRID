static void do_downmix(AC3DecodeContext *ctx)

{

    int from = ctx->bsi.acmod;

    int to = ctx->output;



    switch (from) {

        case AC3_INPUT_DUALMONO:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_dualmono_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO: /* We Assume that sum of both mono channels is requested */

                    mix_dualmono_to_stereo(ctx);

                    break;

            }

            break;

        case AC3_INPUT_MONO:

            switch (to) {

                case AC3_OUTPUT_STEREO:

                    upmix_mono_to_stereo(ctx);

                    break;

            }

            break;

        case AC3_INPUT_STEREO:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_stereo_to_mono(ctx);

                    break;

            }

            break;

        case AC3_INPUT_3F:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_3f_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO:

                    mix_3f_to_stereo(ctx);

                    break;

            }

            break;

        case AC3_INPUT_2F_1R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_2f_1r_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO:

                    mix_2f_1r_to_stereo(ctx);

                    break;

                case AC3_OUTPUT_DOLBY:

                    mix_2f_1r_to_dolby(ctx);

                    break;

            }

            break;

        case AC3_INPUT_3F_1R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_3f_1r_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO:

                    mix_3f_1r_to_stereo(ctx);

                    break;

                case AC3_OUTPUT_DOLBY:

                    mix_3f_1r_to_dolby(ctx);

                    break;

            }

            break;

        case AC3_INPUT_2F_2R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_2f_2r_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO:

                    mix_2f_2r_to_stereo(ctx);

                    break;

                case AC3_OUTPUT_DOLBY:

                    mix_2f_2r_to_dolby(ctx);

                    break;

            }

            break;

        case AC3_INPUT_3F_2R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    mix_3f_2r_to_mono(ctx);

                    break;

                case AC3_OUTPUT_STEREO:

                    mix_3f_2r_to_stereo(ctx);

                    break;

                case AC3_OUTPUT_DOLBY:

                    mix_3f_2r_to_dolby(ctx);

                    break;

            }

            break;

    }

}
