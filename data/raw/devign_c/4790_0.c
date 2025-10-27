static void ffm_set_write_index(AVFormatContext *s, int64_t pos,

                                int64_t file_size)

{

    FFMContext *ffm = s->priv_data;

    ffm->write_index = pos;

    ffm->file_size = file_size;

}
