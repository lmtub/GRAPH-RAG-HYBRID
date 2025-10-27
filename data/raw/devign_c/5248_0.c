static int vorbis_parse_setup_hdr_modes(vorbis_context *vc) {

    GetBitContext *gb=&vc->gb;

    uint_fast8_t i;



    vc->mode_count=get_bits(gb, 6)+1;

    vc->modes=(vorbis_mode *)av_mallocz(vc->mode_count * sizeof(vorbis_mode));



    AV_DEBUG(" There are %d modes.\n", vc->mode_count);



    for(i=0;i<vc->mode_count;++i) {

        vorbis_mode *mode_setup=&vc->modes[i];



        mode_setup->blockflag=get_bits1(gb);

        mode_setup->windowtype=get_bits(gb, 16); //FIXME check

        mode_setup->transformtype=get_bits(gb, 16); //FIXME check

        mode_setup->mapping=get_bits(gb, 8); //FIXME check



        AV_DEBUG(" %d mode: blockflag %d, windowtype %d, transformtype %d, mapping %d \n", i, mode_setup->blockflag, mode_setup->windowtype, mode_setup->transformtype, mode_setup->mapping);

    }

    return 0;

}
