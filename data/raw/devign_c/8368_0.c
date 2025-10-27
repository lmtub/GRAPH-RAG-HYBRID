static void end_frame(AVFilterLink *link)

{

    DeshakeContext *deshake = link->dst->priv;

    AVFilterBufferRef *in  = link->cur_buf;

    AVFilterBufferRef *out = link->dst->outputs[0]->out_buf;

    Transform t;

    float matrix[9];

    float alpha = 2.0 / deshake->refcount;

    char tmp[256];

    Transform orig;



    if (deshake->cx < 0 || deshake->cy < 0 || deshake->cw < 0 || deshake->ch < 0) {

        // Find the most likely global motion for the current frame

        find_motion(deshake, (deshake->ref == NULL) ? in->data[0] : deshake->ref->data[0], in->data[0], link->w, link->h, in->linesize[0], &t);

    } else {

        uint8_t *src1 = (deshake->ref == NULL) ? in->data[0] : deshake->ref->data[0];

        uint8_t *src2 = in->data[0];



        deshake->cx = FFMIN(deshake->cx, link->w);

        deshake->cy = FFMIN(deshake->cy, link->h);



        if ((unsigned)deshake->cx + (unsigned)deshake->cw > link->w) deshake->cw = link->w - deshake->cx;

        if ((unsigned)deshake->cy + (unsigned)deshake->ch > link->h) deshake->ch = link->h - deshake->cy;



        // Quadword align right margin

        deshake->cw &= ~15;



        src1 += deshake->cy * in->linesize[0] + deshake->cx;

        src2 += deshake->cy * in->linesize[0] + deshake->cx;



        find_motion(deshake, src1, src2, deshake->cw, deshake->ch, in->linesize[0], &t);

    }





    // Copy transform so we can output it later to compare to the smoothed value

    orig.vector.x = t.vector.x;

    orig.vector.y = t.vector.y;

    orig.angle = t.angle;

    orig.zoom = t.zoom;



    // Generate a one-sided moving exponential average

    deshake->avg.vector.x = alpha * t.vector.x + (1.0 - alpha) * deshake->avg.vector.x;

    deshake->avg.vector.y = alpha * t.vector.y + (1.0 - alpha) * deshake->avg.vector.y;

    deshake->avg.angle = alpha * t.angle + (1.0 - alpha) * deshake->avg.angle;

    deshake->avg.zoom = alpha * t.zoom + (1.0 - alpha) * deshake->avg.zoom;



    // Remove the average from the current motion to detect the motion that

    // is not on purpose, just as jitter from bumping the camera

    t.vector.x -= deshake->avg.vector.x;

    t.vector.y -= deshake->avg.vector.y;

    t.angle -= deshake->avg.angle;

    t.zoom -= deshake->avg.zoom;



    // Invert the motion to undo it

    t.vector.x *= -1;

    t.vector.y *= -1;

    t.angle *= -1;



    // Write statistics to file

    if (deshake->fp) {

        snprintf(tmp, 256, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", orig.vector.x, deshake->avg.vector.x, t.vector.x, orig.vector.y, deshake->avg.vector.y, t.vector.y, orig.angle, deshake->avg.angle, t.angle, orig.zoom, deshake->avg.zoom, t.zoom);

        fwrite(tmp, sizeof(char), strlen(tmp), deshake->fp);

    }



    // Turn relative current frame motion into absolute by adding it to the

    // last absolute motion

    t.vector.x += deshake->last.vector.x;

    t.vector.y += deshake->last.vector.y;

    t.angle += deshake->last.angle;

    t.zoom += deshake->last.zoom;



    // Shrink motion by 10% to keep things centered in the camera frame

    t.vector.x *= 0.9;

    t.vector.y *= 0.9;

    t.angle *= 0.9;



    // Store the last absolute motion information

    deshake->last.vector.x = t.vector.x;

    deshake->last.vector.y = t.vector.y;

    deshake->last.angle = t.angle;

    deshake->last.zoom = t.zoom;



    // Generate a luma transformation matrix

    avfilter_get_matrix(t.vector.x, t.vector.y, t.angle, 1.0 + t.zoom / 100.0, matrix);



    // Transform the luma plane

    avfilter_transform(in->data[0], out->data[0], in->linesize[0], out->linesize[0], link->w, link->h, matrix, INTERPOLATE_BILINEAR, deshake->edge);



    // Generate a chroma transformation matrix

    avfilter_get_matrix(t.vector.x / (link->w / CHROMA_WIDTH(link)), t.vector.y / (link->h / CHROMA_HEIGHT(link)), t.angle, 1.0 + t.zoom / 100.0, matrix);



    // Transform the chroma planes

    avfilter_transform(in->data[1], out->data[1], in->linesize[1], out->linesize[1], CHROMA_WIDTH(link), CHROMA_HEIGHT(link), matrix, INTERPOLATE_BILINEAR, deshake->edge);

    avfilter_transform(in->data[2], out->data[2], in->linesize[2], out->linesize[2], CHROMA_WIDTH(link), CHROMA_HEIGHT(link), matrix, INTERPOLATE_BILINEAR, deshake->edge);



    // Store the current frame as the reference frame for calculating the

    // motion of the next frame

    if (deshake->ref != NULL)

        avfilter_unref_buffer(deshake->ref);



    // Cleanup the old reference frame

    deshake->ref = in;



    // Draw the transformed frame information

    avfilter_draw_slice(link->dst->outputs[0], 0, link->h, 1);

    avfilter_end_frame(link->dst->outputs[0]);

    avfilter_unref_buffer(out);

}
