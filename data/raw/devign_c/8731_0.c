AVFilterFormats *avfilter_make_all_channel_layouts(void)

{

    static int64_t chlayouts[] = {

        AV_CH_LAYOUT_MONO,

        AV_CH_LAYOUT_STEREO,

        AV_CH_LAYOUT_4POINT0,

        AV_CH_LAYOUT_QUAD,

        AV_CH_LAYOUT_5POINT0,

        AV_CH_LAYOUT_5POINT0_BACK,

        AV_CH_LAYOUT_5POINT1,

        AV_CH_LAYOUT_5POINT1_BACK,

        AV_CH_LAYOUT_5POINT1|AV_CH_LAYOUT_STEREO_DOWNMIX,

        AV_CH_LAYOUT_7POINT1,

        AV_CH_LAYOUT_7POINT1_WIDE,

        AV_CH_LAYOUT_7POINT1|AV_CH_LAYOUT_STEREO_DOWNMIX,

        -1,

    };



    return avfilter_make_format64_list(chlayouts);

}
