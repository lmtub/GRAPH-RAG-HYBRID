offset_t url_filesize(URLContext *h)

{

    offset_t pos, size;



    size= url_seek(h, 0, AVSEEK_SIZE);

    if(size<0){

        pos = url_seek(h, 0, SEEK_CUR);

        size = url_seek(h, -1, SEEK_END)+1;

        url_seek(h, pos, SEEK_SET);

    }

    return size;

}
