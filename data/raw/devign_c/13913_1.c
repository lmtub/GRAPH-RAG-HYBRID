unsigned avutil_version(void)
{
    av_assert0(AV_PIX_FMT_VDA_VLD == 81); //check if the pix fmt enum has not had anything inserted or removed by mistake
    av_assert0(AV_SAMPLE_FMT_DBLP == 9);
    av_assert0(AVMEDIA_TYPE_ATTACHMENT == 4);
    av_assert0(AV_PICTURE_TYPE_BI == 7);
    av_assert0(LIBAVUTIL_VERSION_MICRO >= 100);
    av_assert0(HAVE_MMX2 == HAVE_MMXEXT);
    return LIBAVUTIL_VERSION_INT;