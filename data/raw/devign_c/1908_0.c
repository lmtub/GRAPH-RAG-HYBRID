static int rm_assemble_video_frame(AVFormatContext *s, RMContext *rm, AVPacket *pkt, int len)

{

    ByteIOContext *pb = &s->pb;

    int hdr, seq, pic_num, len2, pos;

    int type;

    int ssize;



    hdr = get_byte(pb); len--;

    type = hdr >> 6;

    switch(type){

    case 0: // slice

    case 2: // last slice

        seq = get_byte(pb); len--;

        len2 = get_num(pb, &len);

        pos = get_num(pb, &len);

        pic_num = get_byte(pb); len--;

        rm->remaining_len = len;

        break;

    case 1: //whole frame

        seq = get_byte(pb); len--;

        if(av_new_packet(pkt, len + 9) < 0)

            return AVERROR(EIO);

        pkt->data[0] = 0;

        AV_WL32(pkt->data + 1, 1);

        AV_WL32(pkt->data + 5, 0);

        get_buffer(pb, pkt->data + 9, len);

        rm->remaining_len = 0;

        return 0;

    case 3: //frame as a part of packet

        len2 = get_num(pb, &len);

        pos = get_num(pb, &len);

        pic_num = get_byte(pb); len--;

        rm->remaining_len = len - len2;

        if(av_new_packet(pkt, len2 + 9) < 0)

            return AVERROR(EIO);

        pkt->data[0] = 0;

        AV_WL32(pkt->data + 1, 1);

        AV_WL32(pkt->data + 5, 0);

        get_buffer(pb, pkt->data + 9, len2);

        return 0;

    }

    //now we have to deal with single slice



    if((seq & 0x7F) == 1 || rm->curpic_num != pic_num){

        rm->slices = ((hdr & 0x3F) << 1) + 1;

        ssize = len2 + 8*rm->slices + 1;

        rm->videobuf = av_realloc(rm->videobuf, ssize);

        rm->videobufsize = ssize;

        rm->videobufpos = 8*rm->slices + 1;

        rm->cur_slice = 0;

        rm->curpic_num = pic_num;

        rm->pktpos = url_ftell(pb);

    }

    if(type == 2){

        len = FFMIN(len, pos);

        pos = len2 - pos;

    }



    if(++rm->cur_slice > rm->cur_slice)

        return 1;

    AV_WL32(rm->videobuf - 7 + 8*rm->cur_slice, 1);

    AV_WL32(rm->videobuf - 3 + 8*rm->cur_slice, rm->videobufpos - 8*rm->slices - 1);

    if(rm->videobufpos + len > rm->videobufsize)

        return 1;

    if (get_buffer(pb, rm->videobuf + rm->videobufpos, len) != len)

        return AVERROR(EIO);

    rm->videobufpos += len,

    rm->remaining_len-= len;



    if(type == 2 || (rm->videobufpos) == rm->videobufsize){

         //adjust slice headers

         memmove(rm->videobuf + 1 + 8*rm->cur_slice, rm->videobuf + 1 + 8*rm->slices, rm->videobufsize - 1 - 8*rm->slices);

         ssize = rm->videobufsize - 8*(rm->slices - rm->cur_slice);



         rm->videobuf[0] = rm->cur_slice-1;

         if(av_new_packet(pkt, ssize) < 0)

             return AVERROR(ENOMEM);

         memcpy(pkt->data, rm->videobuf, ssize);

         pkt->pts = AV_NOPTS_VALUE;

         pkt->pos = rm->pktpos;

         return 0;

    }



    return 1;

}
