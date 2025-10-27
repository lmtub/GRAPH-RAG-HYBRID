static int mov_seek_fragment(AVFormatContext *s, AVStream *st, int64_t timestamp)

{

    MOVContext *mov = s->priv_data;

    int i, j;



    if (!mov->fragment_index_complete)

        return 0;



    for (i = 0; i < mov->fragment_index_count; i++) {

        if (mov->fragment_index_data[i]->track_id == st->id) {

            MOVFragmentIndex *index = index = mov->fragment_index_data[i];

            for (j = index->item_count - 1; j >= 0; j--) {

                if (index->items[j].time <= timestamp) {

                    if (index->items[j].headers_read)

                        return 0;



                    return mov_switch_root(s, index->items[j].moof_offset);

                }

            }

        }

    }



    return 0;

}
