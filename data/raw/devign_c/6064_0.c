static int open_next_file(AVFormatContext *avf)

{

    ConcatContext *cat = avf->priv_data;

    unsigned fileno = cat->cur_file - cat->files;



    if (cat->cur_file->duration == AV_NOPTS_VALUE)

        cat->cur_file->duration = cat->avf->duration - (cat->cur_file->file_inpoint - cat->cur_file->file_start_time);



    if (++fileno >= cat->nb_files) {

        cat->eof = 1;

        return AVERROR_EOF;

    }

    return open_file(avf, fileno);

}
