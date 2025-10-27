static struct pathelem *add_dir_maybe(struct pathelem *path)

{

    DIR *dir;



    if ((dir = opendir(path->pathname)) != NULL) {

        struct dirent *dirent;



        while ((dirent = readdir(dir)) != NULL) {

            if (!streq(dirent->d_name,".") && !streq(dirent->d_name,"..")){

                path = add_entry(path, dirent->d_name);

            }

        }

        closedir(dir);

    }

    return path;

}
