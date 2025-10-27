int pit_get_initial_count(PITState *pit, int channel)

{

    PITChannelState *s = &pit->channels[channel];

    return s->count;

}
