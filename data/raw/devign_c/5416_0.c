static void start_children(FFStream *feed)

{

    if (no_launch)

        return;



    for (; feed; feed = feed->next) {

        if (feed->child_argv && !feed->pid) {

            feed->pid_start = time(0);



            feed->pid = fork();



            if (feed->pid < 0) {

                http_log("Unable to create children\n");

                exit(1);

            }

            if (!feed->pid) {

                /* In child */

                char pathname[1024];

                char *slash;

                int i;



                av_strlcpy(pathname, my_program_name, sizeof(pathname));



                slash = strrchr(pathname, '/');

                if (!slash)

                    slash = pathname;

                else

                    slash++;

                strcpy(slash, "ffmpeg");



                http_log("Launch command line: ");

                http_log("%s ", pathname);

                for (i = 1; feed->child_argv[i] && feed->child_argv[i][0]; i++)

                    http_log("%s ", feed->child_argv[i]);

                http_log("\n");



                for (i = 3; i < 256; i++)

                    close(i);



                if (!ffserver_debug) {

                    i = open("/dev/null", O_RDWR);

                    if (i != -1) {

                        dup2(i, 0);

                        dup2(i, 1);

                        dup2(i, 2);

                        close(i);

                    }

                }



                /* This is needed to make relative pathnames work */

                chdir(my_program_dir);



                signal(SIGPIPE, SIG_DFL);



                execvp(pathname, feed->child_argv);



                _exit(1);

            }

        }

    }

}
