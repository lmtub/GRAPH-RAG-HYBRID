static int ftp_parse_entry_mlsd(char *mlsd, AVIODirEntry *next)

{

    char *fact, *value;

    av_dlog(NULL, "%s\n", mlsd);

    while(fact = av_strtok(mlsd, ";", &mlsd)) {

        if (fact[0] == ' ') {

            next->name = av_strdup(&fact[1]);

            continue;

        }

        fact = av_strtok(fact, "=", &value);

        if (!av_strcasecmp(fact, "type")) {

            if (!av_strcasecmp(value, "cdir") || !av_strcasecmp(value, "pdir"))

                return 1;

            if (!av_strcasecmp(value, "dir"))

                next->type = AVIO_ENTRY_DIRECTORY;

            else if (!av_strcasecmp(value, "file"))

                next->type = AVIO_ENTRY_FILE;

            else if (!av_strcasecmp(value, "OS.unix=slink:"))

                next->type = AVIO_ENTRY_SYMBOLIC_LINK;

        } else if (!av_strcasecmp(fact, "modify")) {

            next->modification_timestamp = ftp_parse_date(value);

        } else if (!av_strcasecmp(fact, "UNIX.mode")) {

            next->filemode = strtoumax(value, NULL, 8);

        } else if (!av_strcasecmp(fact, "UNIX.uid") || !av_strcasecmp(fact, "UNIX.owner"))

            next->user_id = strtoumax(value, NULL, 10);

        else if (!av_strcasecmp(fact, "UNIX.gid") || !av_strcasecmp(fact, "UNIX.group"))

            next->group_id = strtoumax(value, NULL, 10);

        else if (!av_strcasecmp(fact, "size") || !av_strcasecmp(fact, "sizd"))

            next->size = strtoll(value, NULL, 10);

    }

    return 0;

}
