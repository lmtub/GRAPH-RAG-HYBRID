static int select_input_file(uint8_t *no_packet)

{

    int64_t ipts_min = INT64_MAX;

    int i, file_index = -1;



    for (i = 0; i < nb_input_streams; i++) {

        InputStream *ist = input_streams[i];

        int64_t ipts     = ist->pts;



        if (ist->discard || no_packet[ist->file_index])

            continue;

        if (!input_files[ist->file_index]->eof_reached) {

            if (ipts < ipts_min) {

                ipts_min = ipts;

                file_index = ist->file_index;

            }

        }

    }



    return file_index;

}
