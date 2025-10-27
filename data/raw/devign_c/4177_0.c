static bool get_vt_profile_level(AVCodecContext *avctx,

                                 CFStringRef    *profile_level_val)

{

    VTEncContext *vtctx = avctx->priv_data;

    int64_t profile = vtctx->profile;



    if (profile == H264_PROF_AUTO && vtctx->level) {

        //Need to pick a profile if level is not auto-selected.

        profile = vtctx->has_b_frames ? H264_PROF_MAIN : H264_PROF_BASELINE;

    }



    *profile_level_val = NULL;



    switch (profile) {

        case H264_PROF_AUTO:

            return true;



        case H264_PROF_BASELINE:

            switch (vtctx->level) {

                case  0: *profile_level_val = kVTProfileLevel_H264_Baseline_AutoLevel; break;

                case 13: *profile_level_val = kVTProfileLevel_H264_Baseline_1_3;       break;

                case 30: *profile_level_val = kVTProfileLevel_H264_Baseline_3_0;       break;

                case 31: *profile_level_val = kVTProfileLevel_H264_Baseline_3_1;       break;

                case 32: *profile_level_val = kVTProfileLevel_H264_Baseline_3_2;       break;

                case 40: *profile_level_val = kVTProfileLevel_H264_Baseline_4_0;       break;

                case 41: *profile_level_val = kVTProfileLevel_H264_Baseline_4_1;       break;

                case 42: *profile_level_val = kVTProfileLevel_H264_Baseline_4_2;       break;

                case 50: *profile_level_val = kVTProfileLevel_H264_Baseline_5_0;       break;

                case 51: *profile_level_val = kVTProfileLevel_H264_Baseline_5_1;       break;

                case 52: *profile_level_val = kVTProfileLevel_H264_Baseline_5_2;       break;

            }

            break;



        case H264_PROF_MAIN:

            switch (vtctx->level) {

                case  0: *profile_level_val = kVTProfileLevel_H264_Main_AutoLevel; break;

                case 30: *profile_level_val = kVTProfileLevel_H264_Main_3_0;       break;

                case 31: *profile_level_val = kVTProfileLevel_H264_Main_3_1;       break;

                case 32: *profile_level_val = kVTProfileLevel_H264_Main_3_2;       break;

                case 40: *profile_level_val = kVTProfileLevel_H264_Main_4_0;       break;

                case 41: *profile_level_val = kVTProfileLevel_H264_Main_4_1;       break;

                case 42: *profile_level_val = kVTProfileLevel_H264_Main_4_2;       break;

                case 50: *profile_level_val = kVTProfileLevel_H264_Main_5_0;       break;

                case 51: *profile_level_val = kVTProfileLevel_H264_Main_5_1;       break;

                case 52: *profile_level_val = kVTProfileLevel_H264_Main_5_2;       break;

            }

            break;



        case H264_PROF_HIGH:

            switch (vtctx->level) {

                case  0: *profile_level_val = kVTProfileLevel_H264_High_AutoLevel; break;

                case 30: *profile_level_val = kVTProfileLevel_H264_High_3_0;       break;

                case 31: *profile_level_val = kVTProfileLevel_H264_High_3_1;       break;

                case 32: *profile_level_val = kVTProfileLevel_H264_High_3_2;       break;

                case 40: *profile_level_val = kVTProfileLevel_H264_High_4_0;       break;

                case 41: *profile_level_val = kVTProfileLevel_H264_High_4_1;       break;

                case 42: *profile_level_val = kVTProfileLevel_H264_High_4_2;       break;

                case 50: *profile_level_val = kVTProfileLevel_H264_High_5_0;       break;

                case 51: *profile_level_val = kVTProfileLevel_H264_High_5_1;       break;

                case 52: *profile_level_val = kVTProfileLevel_H264_High_5_2;       break;

            }

            break;

    }



    if (!*profile_level_val) {

        av_log(avctx, AV_LOG_ERROR, "Invalid Profile/Level.\n");

        return false;

    }



    return true;

}
