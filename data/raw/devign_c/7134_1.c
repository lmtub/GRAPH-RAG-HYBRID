static int decode_residuals(FLACContext *s, int channel, int pred_order)

{

    int i, tmp, partition, method_type, rice_order;

    int sample = 0, samples;



    method_type = get_bits(&s->gb, 2);

    if (method_type != 0){

        av_log(s->avctx, AV_LOG_DEBUG, "illegal residual coding method %d\n", method_type);





    rice_order = get_bits(&s->gb, 4);



    samples= s->blocksize >> rice_order;







    sample=

    i= pred_order;

    for (partition = 0; partition < (1 << rice_order); partition++)

    {

        tmp = get_bits(&s->gb, 4);

        if (tmp == 15)

        {

            av_log(s->avctx, AV_LOG_DEBUG, "fixed len partition\n");

            tmp = get_bits(&s->gb, 5);

            for (; i < samples; i++, sample++)

                s->decoded[channel][sample] = get_sbits(&s->gb, tmp);


        else

        {

//            av_log(s->avctx, AV_LOG_DEBUG, "rice coded partition k=%d\n", tmp);

            for (; i < samples; i++, sample++){

                s->decoded[channel][sample] = get_sr_golomb_flac(&s->gb, tmp, INT_MAX, 0);



        i= 0;




//    av_log(s->avctx, AV_LOG_DEBUG, "partitions: %d, samples: %d\n", 1 << rice_order, sample);



    return 0;
