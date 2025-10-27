static int get_cv_transfer_function(AVCodecContext *avctx,

                                    CFStringRef *transfer_fnc,

                                    CFNumberRef *gamma_level)

{

    enum AVColorTransferCharacteristic trc = avctx->color_trc;

    Float32 gamma;

    *gamma_level = NULL;



    switch (trc) {

        case AVCOL_TRC_UNSPECIFIED:

            *transfer_fnc = NULL;

            break;



        case AVCOL_TRC_BT709:

            *transfer_fnc = kCVImageBufferTransferFunction_ITU_R_709_2;

            break;



        case AVCOL_TRC_SMPTE240M:

            *transfer_fnc = kCVImageBufferTransferFunction_SMPTE_240M_1995;

            break;



        case AVCOL_TRC_GAMMA22:

            gamma = 2.2;

            *transfer_fnc = kCVImageBufferTransferFunction_UseGamma;

            *gamma_level = CFNumberCreate(NULL, kCFNumberFloat32Type, &gamma);

            break;



        case AVCOL_TRC_GAMMA28:

            gamma = 2.8;

            *transfer_fnc = kCVImageBufferTransferFunction_UseGamma;

            *gamma_level = CFNumberCreate(NULL, kCFNumberFloat32Type, &gamma);

            break;



        case AVCOL_TRC_BT2020_10:

        case AVCOL_TRC_BT2020_12:

            *transfer_fnc = kCVImageBufferTransferFunction_ITU_R_2020;

            break;



        default:

            av_log(avctx, AV_LOG_ERROR, "Transfer function %s is not supported.\n", av_color_transfer_name(trc));

            return -1;

    }



    return 0;

}
