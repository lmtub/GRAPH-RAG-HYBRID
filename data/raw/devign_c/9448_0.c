static const char *ass_split_section(ASSSplitContext *ctx, const char *buf)

{

    const ASSSection *section = &ass_sections[ctx->current_section];

    int *number = &ctx->field_number[ctx->current_section];

    int *order = ctx->field_order[ctx->current_section];

    int *tmp, i, len;



    while (buf && *buf) {

        if (buf[0] == '[') {

            ctx->current_section = -1;

            break;

        }

        if (buf[0] == ';' || (buf[0] == '!' && buf[1] == ':')) {

            /* skip comments */

        } else if (section->format_header && !order) {

            len = strlen(section->format_header);

            if (strncmp(buf, section->format_header, len) || buf[len] != ':')

                return NULL;

            buf += len + 1;

            while (!is_eol(*buf)) {

                buf = skip_space(buf);

                len = strcspn(buf, ", \r\n");

                if (!(tmp = av_realloc(order, (*number + 1) * sizeof(*order))))

                    return NULL;

                order = tmp;

                order[*number] = -1;

                for (i=0; section->fields[i].name; i++)

                    if (!strncmp(buf, section->fields[i].name, len)) {

                        order[*number] = i;

                        break;

                    }

                (*number)++;

                buf = skip_space(buf + len + (buf[len] == ','));

            }

            ctx->field_order[ctx->current_section] = order;

        } else if (section->fields_header) {

            len = strlen(section->fields_header);

            if (!strncmp(buf, section->fields_header, len) && buf[len] == ':') {

                uint8_t *ptr, *struct_ptr = realloc_section_array(ctx);

                if (!struct_ptr)  return NULL;

                buf += len + 1;

                for (i=0; !is_eol(*buf) && i < *number; i++) {

                    int last = i == *number - 1;

                    buf = skip_space(buf);

                    len = strcspn(buf, last ? "\r\n" : ",\r\n");

                    if (order[i] >= 0) {

                        ASSFieldType type = section->fields[order[i]].type;

                        ptr = struct_ptr + section->fields[order[i]].offset;

                        convert_func[type](ptr, buf, len);

                    }

                    buf = skip_space(buf + len + !last);

                }

            }

        } else {

            len = strcspn(buf, ":\r\n");

            if (buf[len] == ':') {

                for (i=0; section->fields[i].name; i++)

                    if (!strncmp(buf, section->fields[i].name, len)) {

                        ASSFieldType type = section->fields[i].type;

                        uint8_t *ptr = (uint8_t *)&ctx->ass + section->offset;

                        ptr += section->fields[i].offset;

                        buf = skip_space(buf + len + 1);

                        convert_func[type](ptr, buf, strcspn(buf, "\r\n"));

                        break;

                    }

            }

        }

        buf += strcspn(buf, "\n");

        buf += !!*buf;

    }

    return buf;

}
