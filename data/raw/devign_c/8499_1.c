libAVMemInputPin_Receive(libAVMemInputPin *this, IMediaSample *sample)

{

    libAVPin *pin = (libAVPin *) ((uint8_t *) this - imemoffset);

    enum dshowDeviceType devtype = pin->filter->type;

    void *priv_data;

    uint8_t *buf;

    int buf_size;

    int index;

    int64_t curtime;



    dshowdebug("libAVMemInputPin_Receive(%p)\n", this);



    if (!sample)

        return E_POINTER;



    if (devtype == VideoDevice) {

        /* PTS from video devices is unreliable. */

        IReferenceClock *clock = pin->filter->clock;

        IReferenceClock_GetTime(clock, &curtime);

    } else {

        int64_t dummy;

        IMediaSample_GetTime(sample, &curtime, &dummy);

        curtime += pin->filter->start_time;

    }



    buf_size = IMediaSample_GetActualDataLength(sample);

    IMediaSample_GetPointer(sample, &buf);

    priv_data = pin->filter->priv_data;

    index = pin->filter->stream_index;



    pin->filter->callback(priv_data, index, buf, buf_size, curtime);



    return S_OK;

}
