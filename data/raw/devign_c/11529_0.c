static av_cold int check_cuda_errors(AVCodecContext *avctx, CUresult err, const char *func)

{

    if (err != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, ">> %s - failed with error code 0x%x\n", func, err);

        return 0;

    }

    return 1;

}
