int pit_get_gate(PITState *pit, int channel)

{

    PITChannelState *s = &pit->channels[channel];

    return s->gate;

}
