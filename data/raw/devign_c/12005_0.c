static int mov_read_seek(AVFormatContext *s, int stream_index, int64_t sample_time, int flags)

{

    MOVContext* mov = (MOVContext *) s->priv_data;

    MOVStreamContext* sc;

    int32_t i, a, b, m;

    int64_t start_time;

    int32_t seek_sample, sample;

    int32_t duration;

    int32_t count;

    int32_t chunk;

    int32_t left_in_chunk;

    int64_t chunk_file_offset;

    int64_t sample_file_offset;

    int32_t first_chunk_sample;

    int32_t sample_to_chunk_idx;

    int sample_to_time_index;

    long sample_to_time_sample = 0;

    uint64_t sample_to_time_time = 0;

    int mov_idx;



    // Find the corresponding mov stream

    for (mov_idx = 0; mov_idx < mov->total_streams; mov_idx++)

        if (mov->streams[mov_idx]->ffindex == stream_index)

            break;

    if (mov_idx == mov->total_streams) {

        av_log(s, AV_LOG_ERROR, "mov: requested stream was not found in mov streams (idx=%i)\n", stream_index);

        return -1;

    }

    sc = mov->streams[mov_idx];



    sample_time *= s->streams[stream_index]->time_base.num;



    // Step 1. Find the edit that contains the requested time (elst)

    if (sc->edit_count && 0) {

        // FIXME should handle edit list

        av_log(s, AV_LOG_ERROR, "mov: does not handle seeking in files that contain edit list (c:%d)\n", sc->edit_count);

        return -1;

    }



    // Step 2. Find the corresponding sample using the Time-to-sample atom (stts) */

    dprintf("Searching for time %li in stream #%i (time_scale=%i)\n", (long)sample_time, mov_idx, sc->time_scale);

    start_time = 0; // FIXME use elst atom

    sample = 1; // sample are 0 based in table



    for (i = 0; i < sc->stts_count; i++) {

        count = sc->stts_data[i].count;

        duration = sc->stts_data[i].duration;

        if ((start_time + count*duration) > sample_time) {

            sample_to_time_time = start_time;

            sample_to_time_index = i;

            sample_to_time_sample = sample;

            sample += (sample_time - start_time) / duration;

            break;

        }

        sample += count;

        start_time += count * duration;

    }

    sample_to_time_time = start_time;

    sample_to_time_index = i;

    /* NOTE: despite what qt doc say, the dt value (Display Time in qt vocabulary) computed with the stts atom

       is a decoding time stamp (dts) not a presentation time stamp. And as usual dts != pts for stream with b frames */



    dprintf("Found time %li at sample #%u\n", (long)sample_time, sample);

    if (sample > sc->sample_count) {

        av_log(s, AV_LOG_ERROR, "mov: sample pos is too high, unable to seek (req. sample=%i, sample count=%ld)\n", sample, sc->sample_count);

        return -1;

    }



    // Step 3. Find the prior sync. sample using the Sync sample atom (stss)

    if (sc->keyframes) {

        a = 0;

        b = sc->keyframe_count - 1;

        while (a < b) {

            m = (a + b + 1) >> 1;

            if (sc->keyframes[m] > sample) {

                b = m - 1;

            } else {

                a = m;

            }

        }

        // for low latency prob: always use the previous keyframe, just uncomment the next line

        // if (a) a--;

        seek_sample = sc->keyframes[a];

    }

    else

        seek_sample = sample; // else all samples are key frames

    dprintf("Found nearest keyframe at sample #%i \n", seek_sample);



    // Step 4. Find the chunk of the sample using the Sample-to-chunk-atom (stsc)

    for (first_chunk_sample = 1, i = 0; i < (sc->sample_to_chunk_sz - 1); i++) {

        b = (sc->sample_to_chunk[i + 1].first - sc->sample_to_chunk[i].first) * sc->sample_to_chunk[i].count;

        if (seek_sample >= first_chunk_sample && seek_sample < (first_chunk_sample + b))

            break;

        first_chunk_sample += b;

    }

    chunk = sc->sample_to_chunk[i].first + (seek_sample - first_chunk_sample) / sc->sample_to_chunk[i].count;

    left_in_chunk = sc->sample_to_chunk[i].count - (seek_sample - first_chunk_sample) % sc->sample_to_chunk[i].count;

    first_chunk_sample += ((seek_sample - first_chunk_sample) / sc->sample_to_chunk[i].count) * sc->sample_to_chunk[i].count;

    sample_to_chunk_idx = i;

    dprintf("Sample was found in chunk #%i at sample offset %i (idx %i)\n", chunk, seek_sample - first_chunk_sample, sample_to_chunk_idx);



    // Step 5. Find the offset of the chunk using the chunk offset atom

    if (!sc->chunk_offsets) {

        av_log(s, AV_LOG_ERROR, "mov: no chunk offset atom, unable to seek\n");

        return -1;

    }

    if (chunk > sc->chunk_count) {

        av_log(s, AV_LOG_ERROR, "mov: chunk offset atom too short, unable to seek (req. chunk=%i, chunk count=%li)\n", chunk, sc->chunk_count);

        return -1;

    }

    chunk_file_offset = sc->chunk_offsets[chunk - 1];

    dprintf("Chunk file offset is #%"PRIu64"\n", chunk_file_offset);



    // Step 6. Find the byte offset within the chunk using the sample size atom

    sample_file_offset = chunk_file_offset;

    if (sc->sample_size)

        sample_file_offset += (seek_sample - first_chunk_sample) * sc->sample_size;

    else {

        for (i = 0; i < (seek_sample - first_chunk_sample); i++) {

        sample_file_offset += sc->sample_sizes[first_chunk_sample + i - 1];

        }

    }

    dprintf("Sample file offset is #%"PRIu64"\n", sample_file_offset);



    // Step 6. Update the parser

    mov->partial = sc;

    mov->next_chunk_offset = sample_file_offset;

    // Update current stream state

    sc->current_sample = seek_sample - 1;  // zero based

    sc->left_in_chunk = left_in_chunk;

    sc->next_chunk = chunk; // +1 -1 (zero based)

    sc->sample_to_chunk_index = sample_to_chunk_idx;



    // Update other streams

    for (i = 0; i<mov->total_streams; i++) {

        MOVStreamContext *msc;

        if (i == mov_idx) continue;

        // Find the nearest 'next' chunk

        msc = mov->streams[i];

        a = 0;

        b = msc->chunk_count - 1;

        while (a < b) {

            m = (a + b + 1) >> 1;

            if (msc->chunk_offsets[m] > chunk_file_offset) {

                b = m - 1;

            } else {

                a = m;

            }

        }

        msc->next_chunk = a;

        if (msc->chunk_offsets[a] < chunk_file_offset && a < (msc->chunk_count-1))

            msc->next_chunk ++;

        dprintf("Nearest next chunk for stream #%i is #%li @%"PRId64"\n", i, msc->next_chunk+1, msc->chunk_offsets[msc->next_chunk]);



        // Compute sample count and index in the sample_to_chunk table (what a pity)

        msc->sample_to_chunk_index = 0;

        msc->current_sample = 0;

        for(;  msc->sample_to_chunk_index < (msc->sample_to_chunk_sz - 1)

            && msc->sample_to_chunk[msc->sample_to_chunk_index + 1].first <= (1 + msc->next_chunk); msc->sample_to_chunk_index++) {

            msc->current_sample += (msc->sample_to_chunk[msc->sample_to_chunk_index + 1].first - msc->sample_to_chunk[msc->sample_to_chunk_index].first) \

            * msc->sample_to_chunk[msc->sample_to_chunk_index].count;

        }

        msc->current_sample += (msc->next_chunk - (msc->sample_to_chunk[msc->sample_to_chunk_index].first - 1)) * sc->sample_to_chunk[msc->sample_to_chunk_index].count;

        msc->left_in_chunk = msc->sample_to_chunk[msc->sample_to_chunk_index].count - 1;

        // Find corresponding position in stts (used later to compute dts)

        sample = 0;

        start_time = 0;

        for (msc->sample_to_time_index = 0; msc->sample_to_time_index < msc->stts_count; msc->sample_to_time_index++) {

            count = msc->stts_data[msc->sample_to_time_index].count;

            duration = msc->stts_data[msc->sample_to_time_index].duration;

            if ((sample + count - 1) > msc->current_sample) {

                msc->sample_to_time_time = start_time;

                msc->sample_to_time_sample = sample;

                break;

            }

            sample += count;

            start_time += count * duration;

        }

        sample = 0;

        for (msc->sample_to_ctime_index = 0; msc->sample_to_ctime_index < msc->ctts_count; msc->sample_to_ctime_index++) {

            count = msc->ctts_data[msc->sample_to_ctime_index].count;

            duration = msc->ctts_data[msc->sample_to_ctime_index].duration;

            if ((sample + count - 1) > msc->current_sample) {

                msc->sample_to_ctime_sample = sample;

                break;

            }

            sample += count;

        }

        dprintf("Next Sample for stream #%i is #%li @%li\n", i, msc->current_sample + 1, msc->sample_to_chunk_index + 1);

    }

    return 0;

}
