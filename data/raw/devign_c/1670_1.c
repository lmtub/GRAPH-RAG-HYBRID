static int coreaudio_init_out (HWVoiceOut *hw, struct audsettings *as)

{

    OSStatus status;

    coreaudioVoiceOut *core = (coreaudioVoiceOut *) hw;

    UInt32 propertySize;

    int err;

    const char *typ = "playback";

    AudioValueRange frameRange;



    /* create mutex */

    err = pthread_mutex_init(&core->mutex, NULL);

    if (err) {

        dolog("Could not create mutex\nReason: %s\n", strerror (err));

        return -1;

    }



    audio_pcm_init_info (&hw->info, as);



    /* open default output device */

    propertySize = sizeof(core->outputDeviceID);

    status = AudioHardwareGetProperty(

        kAudioHardwarePropertyDefaultOutputDevice,

        &propertySize,

        &core->outputDeviceID);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ,

                           "Could not get default output Device\n");

        return -1;

    }

    if (core->outputDeviceID == kAudioDeviceUnknown) {

        dolog ("Could not initialize %s - Unknown Audiodevice\n", typ);

        return -1;

    }



    /* get minimum and maximum buffer frame sizes */

    propertySize = sizeof(frameRange);

    status = AudioDeviceGetProperty(

        core->outputDeviceID,

        0,

        0,

        kAudioDevicePropertyBufferFrameSizeRange,

        &propertySize,

        &frameRange);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ,

                           "Could not get device buffer frame range\n");

        return -1;

    }



    if (frameRange.mMinimum > conf.buffer_frames) {

        core->audioDevicePropertyBufferFrameSize = (UInt32) frameRange.mMinimum;

        dolog ("warning: Upsizing Buffer Frames to %f\n", frameRange.mMinimum);

    }

    else if (frameRange.mMaximum < conf.buffer_frames) {

        core->audioDevicePropertyBufferFrameSize = (UInt32) frameRange.mMaximum;

        dolog ("warning: Downsizing Buffer Frames to %f\n", frameRange.mMaximum);

    }

    else {

        core->audioDevicePropertyBufferFrameSize = conf.buffer_frames;

    }



    /* set Buffer Frame Size */

    propertySize = sizeof(core->audioDevicePropertyBufferFrameSize);

    status = AudioDeviceSetProperty(

        core->outputDeviceID,

        NULL,

        0,

        false,

        kAudioDevicePropertyBufferFrameSize,

        propertySize,

        &core->audioDevicePropertyBufferFrameSize);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ,

                           "Could not set device buffer frame size %" PRIu32 "\n",

                           (uint32_t)core->audioDevicePropertyBufferFrameSize);

        return -1;

    }



    /* get Buffer Frame Size */

    propertySize = sizeof(core->audioDevicePropertyBufferFrameSize);

    status = AudioDeviceGetProperty(

        core->outputDeviceID,

        0,

        false,

        kAudioDevicePropertyBufferFrameSize,

        &propertySize,

        &core->audioDevicePropertyBufferFrameSize);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ,

                           "Could not get device buffer frame size\n");

        return -1;

    }

    hw->samples = conf.nbuffers * core->audioDevicePropertyBufferFrameSize;



    /* get StreamFormat */

    propertySize = sizeof(core->outputStreamBasicDescription);

    status = AudioDeviceGetProperty(

        core->outputDeviceID,

        0,

        false,

        kAudioDevicePropertyStreamFormat,

        &propertySize,

        &core->outputStreamBasicDescription);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ,

                           "Could not get Device Stream properties\n");

        core->outputDeviceID = kAudioDeviceUnknown;

        return -1;

    }



    /* set Samplerate */

    core->outputStreamBasicDescription.mSampleRate = (Float64) as->freq;

    propertySize = sizeof(core->outputStreamBasicDescription);

    status = AudioDeviceSetProperty(

        core->outputDeviceID,

        0,

        0,

        0,

        kAudioDevicePropertyStreamFormat,

        propertySize,

        &core->outputStreamBasicDescription);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ, "Could not set samplerate %d\n",

                           as->freq);

        core->outputDeviceID = kAudioDeviceUnknown;

        return -1;

    }



    /* set Callback */

    status = AudioDeviceAddIOProc(core->outputDeviceID, audioDeviceIOProc, hw);

    if (status != kAudioHardwareNoError) {

        coreaudio_logerr2 (status, typ, "Could not set IOProc\n");

        core->outputDeviceID = kAudioDeviceUnknown;

        return -1;

    }



    /* start Playback */

    if (!isPlaying(core->outputDeviceID)) {

        status = AudioDeviceStart(core->outputDeviceID, audioDeviceIOProc);

        if (status != kAudioHardwareNoError) {

            coreaudio_logerr2 (status, typ, "Could not start playback\n");

            AudioDeviceRemoveIOProc(core->outputDeviceID, audioDeviceIOProc);

            core->outputDeviceID = kAudioDeviceUnknown;

            return -1;

        }

    }



    return 0;

}
