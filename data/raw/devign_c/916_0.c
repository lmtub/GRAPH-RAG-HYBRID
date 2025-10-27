static void qdm2_decode_fft_packets (QDM2Context *q)

{

    int i, j, min, max, value, type, unknown_flag;

    GetBitContext gb;



    if (q->sub_packet_list_B[0].packet == NULL)

        return;



    /* reset minimum indices for FFT coefficients */

    q->fft_coefs_index = 0;

    for (i=0; i < 5; i++)

        q->fft_coefs_min_index[i] = -1;



    /* process subpackets ordered by type, largest type first */

    for (i = 0, max = 256; i < q->sub_packets_B; i++) {

        QDM2SubPacket *packet;



        /* find subpacket with largest type less than max */

        for (j = 0, min = 0, packet = NULL; j < q->sub_packets_B; j++) {

            value = q->sub_packet_list_B[j].packet->type;

            if (value > min && value < max) {

                min = value;

                packet = q->sub_packet_list_B[j].packet;

            }

        }



        max = min;



        /* check for errors (?) */

        if (i == 0 && (packet->type < 16 || packet->type >= 48 || fft_subpackets[packet->type - 16]))

            return;



        /* decode FFT tones */

        init_get_bits (&gb, packet->data, packet->size*8);



        if (packet->type >= 32 && packet->type < 48 && !fft_subpackets[packet->type - 16])

            unknown_flag = 1;

        else

            unknown_flag = 0;



        type = packet->type;



        if ((type >= 17 && type < 24) || (type >= 33 && type < 40)) {

            int duration = q->sub_sampling + 5 - (type & 15);



            if (duration >= 0 && duration < 4)

                qdm2_fft_decode_tones(q, duration, &gb, unknown_flag);

        } else if (type == 31) {

            for (i=0; i < 4; i++)

                qdm2_fft_decode_tones(q, i, &gb, unknown_flag);

        } else if (type == 46) {

            for (i=0; i < 6; i++)

                q->fft_level_exp[i] = get_bits(&gb, 6);

            for (i=0; i < 4; i++)

            qdm2_fft_decode_tones(q, i, &gb, unknown_flag);

        }

    } // Loop on B packets



    /* calculate maximum indices for FFT coefficients */

    for (i = 0, j = -1; i < 5; i++)

        if (q->fft_coefs_min_index[i] >= 0) {

            if (j >= 0)

                q->fft_coefs_max_index[j] = q->fft_coefs_min_index[i];

            j = i;

        }

    if (j >= 0)

        q->fft_coefs_max_index[j] = q->fft_coefs_index;

}
