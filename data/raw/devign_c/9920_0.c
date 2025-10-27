int ff_framesync_dualinput_get_writable(FFFrameSync *fs, AVFrame **f0, AVFrame **f1)

{

    int ret;



    ret = ff_framesync_dualinput_get(fs, f0, f1);

    if (ret < 0)

        return ret;

    ret = ff_inlink_make_frame_writable(fs->parent->inputs[0], f0);

    if (ret < 0) {

        av_frame_free(f0);

        av_frame_free(f1);

        return ret;

    }

    return 0;

}
