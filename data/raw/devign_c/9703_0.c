static int intel_hda_send_command(IntelHDAState *d, uint32_t verb)

{

    uint32_t cad, nid, data;

    HDACodecDevice *codec;

    HDACodecDeviceClass *cdc;



    cad = (verb >> 28) & 0x0f;

    if (verb & (1 << 27)) {

        /* indirect node addressing, not specified in HDA 1.0 */

        dprint(d, 1, "%s: indirect node addressing (guest bug?)\n", __FUNCTION__);

        return -1;

    }

    nid = (verb >> 20) & 0x7f;

    data = verb & 0xfffff;



    codec = hda_codec_find(&d->codecs, cad);

    if (codec == NULL) {

        dprint(d, 1, "%s: addressed non-existing codec\n", __FUNCTION__);

        return -1;

    }

    cdc = HDA_CODEC_DEVICE_GET_CLASS(codec);

    cdc->command(codec, nid, data);

    return 0;

}
