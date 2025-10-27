static int nut_write_trailer(AVFormatContext *s)

{

    NUTContext *nut = s->priv_data;

    ByteIOContext *bc = &s->pb;



    update_packetheader(nut, bc, 0);



#if 0

    int i;



    /* WRITE INDEX */



    for (i = 0; s->nb_streams; i++)

    {

	put_be64(bc, INDEX_STARTCODE);

	put_packetheader(nut, bc, 64);

	put_v(bc, s->streams[i]->id);

	put_v(bc, ...);

	put_be32(bc, 0); /* FIXME: checksum */

        update_packetheader(nut, bc, 0);

    }

#endif



    put_flush_packet(bc);

    

    av_freep(&nut->stream);



    return 0;

}
