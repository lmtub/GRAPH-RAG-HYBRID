static int mmap_read_frame(struct video_data *s, void *frame, int64_t *ts)

{

    struct v4l2_buffer buf;

    int res;



    memset(&buf, 0, sizeof(struct v4l2_buffer));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    buf.memory = V4L2_MEMORY_MMAP;



    /* FIXME: Some special treatment might be needed in case of loss of signal... */

    while ((res = ioctl(s->fd, VIDIOC_DQBUF, &buf)) < 0 &&

           ((errno == EAGAIN) || (errno == EINTR)));

    if (res < 0) {

        av_log(NULL, AV_LOG_ERROR, "ioctl(VIDIOC_DQBUF): %s\n", strerror(errno));



        return -1;

    }

    assert (buf.index < s->buffers);

    assert(buf.bytesused == s->frame_size);

    /* Image is at s->buff_start[buf.index] */

    memcpy(frame, s->buf_start[buf.index], buf.bytesused);

    *ts = buf.timestamp.tv_sec * int64_t_C(1000000) + buf.timestamp.tv_usec;



    res = ioctl (s->fd, VIDIOC_QBUF, &buf);

    if (res < 0) {

        av_log(NULL, AV_LOG_ERROR, "ioctl(VIDIOC_QBUF)\n");



        return -1;

    }



    return s->buf_len[buf.index];

}
