static av_cold int avisynth_load_library(void)

{

    avs_library.library = LoadLibrary(AVISYNTH_LIB);

    if (!avs_library.library)

        return AVERROR_UNKNOWN;



#define LOAD_AVS_FUNC(name, continue_on_fail)                          \

        avs_library.name =                                             \

            (void *)GetProcAddress(avs_library.library, #name);        \

        if (!continue_on_fail && !avs_library.name)                    \

            goto fail;



    LOAD_AVS_FUNC(avs_bit_blt, 0);

    LOAD_AVS_FUNC(avs_clip_get_error, 0);

    LOAD_AVS_FUNC(avs_create_script_environment, 0);

    LOAD_AVS_FUNC(avs_delete_script_environment, 0);

    LOAD_AVS_FUNC(avs_get_audio, 0);

    LOAD_AVS_FUNC(avs_get_error, 1); // New to AviSynth 2.6

    LOAD_AVS_FUNC(avs_get_frame, 0);

    LOAD_AVS_FUNC(avs_get_version, 0);

    LOAD_AVS_FUNC(avs_get_video_info, 0);

    LOAD_AVS_FUNC(avs_invoke, 0);

    LOAD_AVS_FUNC(avs_release_clip, 0);

    LOAD_AVS_FUNC(avs_release_value, 0);

    LOAD_AVS_FUNC(avs_release_video_frame, 0);

    LOAD_AVS_FUNC(avs_take_clip, 0);

#ifdef USING_AVISYNTH

    LOAD_AVS_FUNC(avs_bits_per_pixel, 1);

    LOAD_AVS_FUNC(avs_get_height_p, 1);

    LOAD_AVS_FUNC(avs_get_pitch_p, 1);

    LOAD_AVS_FUNC(avs_get_read_ptr_p, 1);

    LOAD_AVS_FUNC(avs_get_row_size_p, 1);

    LOAD_AVS_FUNC(avs_is_yv24, 1);

    LOAD_AVS_FUNC(avs_is_yv16, 1);

    LOAD_AVS_FUNC(avs_is_yv411, 1);

    LOAD_AVS_FUNC(avs_is_y8, 1);

#endif

#undef LOAD_AVS_FUNC



    atexit(avisynth_atexit_handler);

    return 0;



fail:

    FreeLibrary(avs_library.library);

    return AVERROR_UNKNOWN;

}
