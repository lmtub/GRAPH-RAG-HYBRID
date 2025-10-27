static inline void RENAME(rgb2rgb_init)(void)

{

#if !COMPILE_TEMPLATE_SSE2

#if !COMPILE_TEMPLATE_AMD3DNOW

    rgb15to16          = RENAME(rgb15to16);

    rgb15tobgr24       = RENAME(rgb15tobgr24);

    rgb15to32          = RENAME(rgb15to32);

    rgb16tobgr24       = RENAME(rgb16tobgr24);

    rgb16to32          = RENAME(rgb16to32);

    rgb16to15          = RENAME(rgb16to15);

    rgb24tobgr16       = RENAME(rgb24tobgr16);

    rgb24tobgr15       = RENAME(rgb24tobgr15);

    rgb24tobgr32       = RENAME(rgb24tobgr32);

    rgb32to16          = RENAME(rgb32to16);

    rgb32to15          = RENAME(rgb32to15);

    rgb32tobgr24       = RENAME(rgb32tobgr24);

    rgb24to15          = RENAME(rgb24to15);

    rgb24to16          = RENAME(rgb24to16);

    rgb24tobgr24       = RENAME(rgb24tobgr24);

    shuffle_bytes_2103 = RENAME(shuffle_bytes_2103);

    rgb32tobgr16       = RENAME(rgb32tobgr16);

    rgb32tobgr15       = RENAME(rgb32tobgr15);

    yv12toyuy2         = RENAME(yv12toyuy2);

    yv12touyvy         = RENAME(yv12touyvy);

    yuv422ptoyuy2      = RENAME(yuv422ptoyuy2);

    yuv422ptouyvy      = RENAME(yuv422ptouyvy);

    yuy2toyv12         = RENAME(yuy2toyv12);

    vu9_to_vu12        = RENAME(vu9_to_vu12);

    yvu9_to_yuy2       = RENAME(yvu9_to_yuy2);

    uyvytoyuv422       = RENAME(uyvytoyuv422);

    yuyvtoyuv422       = RENAME(yuyvtoyuv422);

#endif /* !COMPILE_TEMPLATE_AMD3DNOW */



#if COMPILE_TEMPLATE_MMXEXT || COMPILE_TEMPLATE_AMD3DNOW

    planar2x           = RENAME(planar2x);

#endif /* COMPILE_TEMPLATE_MMXEXT || COMPILE_TEMPLATE_AMD3DNOW */

    rgb24toyv12        = RENAME(rgb24toyv12);



    yuyvtoyuv420       = RENAME(yuyvtoyuv420);

    uyvytoyuv420       = RENAME(uyvytoyuv420);

#endif /* !COMPILE_TEMPLATE_SSE2 */



#if !COMPILE_TEMPLATE_AMD3DNOW

    interleaveBytes    = RENAME(interleaveBytes);

#endif /* !COMPILE_TEMPLATE_AMD3DNOW */

}
