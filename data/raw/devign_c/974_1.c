static int mov_read_mdhd(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];



    print_atom("mdhd", atom);



    get_byte(pb); /* version */



    get_byte(pb); get_byte(pb);

    get_byte(pb); /* flags */



    get_be32(pb); /* creation time */

    get_be32(pb); /* modification time */



    c->streams[c->total_streams]->time_scale = get_be32(pb);



#ifdef DEBUG

    printf("track[%i].time_scale = %i\n", c->fc->nb_streams-1, c->streams[c->total_streams]->time_scale); /* time scale */

#endif

    get_be32(pb); /* duration */



    get_be16(pb); /* language */

    get_be16(pb); /* quality */



    return 0;

}
