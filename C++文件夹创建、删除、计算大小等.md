```c
#include <iostream>
#include <unistd.h>            // access
#include <sys/types.h>         // mkdir   rmdir
#include <sys/stat.h>          // mkdir   rmdir
#include <dirent.h>            // opendir readdir closedir
#include <string.h>
#include <string>
#include <time.h>
#include <vector>
#include <algorithm>

using namespace std;

std::string getCurrentTimeStr(){
  time_t t = time(NULL);
  char ch[64] = {0};
  char result[100] = {0};
  strftime(ch, sizeof(ch) - 1, "%Y%m%d--%H%M%S", localtime(&t));
  sprintf(result, "%s", ch);
  return std::string(result);
}

int CreateFolder(char* path) {
	if(access(path, 6) == 0) {
        cout << path << " exist!\n";
    } else {
    	if(mkdir(path, 0777) == 0) {
            cout << "create " << path << "\n";
            return 1;
        } else {
            cout << "fail to create " << path << endl;
            return -1;
        }
    }
    return 0;
}

long long int GetDirectorySize(char *dir)
{
    DIR *dp;
    struct dirent *entry;
    struct stat statbuf;
    long long int totalSize=0;

    if ((dp = opendir(dir)) == NULL)
    {
        fprintf(stderr, "Cannot open dir: %s\n", dir);
        return -1; //可能是个文件，或者目录不存在
    }
    
    //先加上自身目录的大小
    lstat(dir, &statbuf);
    totalSize+=statbuf.st_size;

    while ((entry = readdir(dp)) != NULL)
    {
        char subdir[256];
        sprintf(subdir, "%s/%s", dir, entry->d_name);
        lstat(subdir, &statbuf);
        
        // 子文件夹递归计算大小，子文件直接加
        if (S_ISDIR(statbuf.st_mode))
        {
            if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0)
            {
                continue;
            }

            long long int subDirSize = GetDirectorySize(subdir);
            totalSize+=subDirSize;
        }
        else
        {
            totalSize+=statbuf.st_size;
        }
    }

    closedir(dp);    
    return totalSize;
}

int FindFileInFolder(char *dir_name, vector<string> &file_names) {
    int number = 0;
    // check the parameter !
    if( NULL == dir_name ) {
        cout << " dir_name is null ! " << endl;
        return 0;
    }
    // check if dir_name is a valid directory
    //struct stat s;
    //lstat( dir_name , &s );

    DIR *dir = opendir( dir_name );
    if( NULL == dir ) {
        cout<<"Can not open dir "<<dir_name<<endl;
        return 0;
    }
    /* read all the files in the dir ~ */
    struct dirent *filename;
    while( ( filename = readdir(dir) ) != NULL ) {
        // get rid of "." and ".."
        if( strcmp( filename->d_name , "." ) == 0 || strcmp( filename->d_name , "..") == 0    )
            continue;
        //获取文件后缀
        string sFilename(filename->d_name);
        file_names.push_back(sFilename);
        ++number;
    }
    closedir(dir);
    return number;
}

int rm_dir(std::string dir_full_path) 
{
    DIR* dirp = opendir(dir_full_path.c_str());
    if(!dirp)
    {
        return -1;
    }
    struct dirent *dir;
    struct stat st;
    while((dir = readdir(dirp)) != NULL)
    {
        if(strcmp(dir->d_name,".") == 0 || strcmp(dir->d_name,"..") == 0)
        {
            continue;
        }
        std::string sub_path = dir_full_path + '/' + dir->d_name;
        if(lstat(sub_path.c_str(), &st) == -1)
        {
            //Log("rm_dir:lstat ",sub_path," error");
            continue;
        }
        if(S_ISDIR(st.st_mode))
        {
            if(rm_dir(sub_path) == -1) // 如果是目录文件，递归删除
            {
                closedir(dirp);
                return -1;
            }
            rmdir(sub_path.c_str());
        }
        else if(S_ISREG(st.st_mode))
        {
            unlink(sub_path.c_str());     // 如果是普通文件，则unlink
        }
        else
        {
            //Log("rm_dir:st_mode ",sub_path," error");
            continue;
        }
    }
    if(rmdir(dir_full_path.c_str()) == -1)//delete dir itself.
    {
        closedir(dirp);
        return -1;
    }
    closedir(dirp);
    return 0;
}

void DeleteRedundancy(char *path, long long thr) {
	vector<string> fns;
	FindFileInFolder(path, fns);
	int n = fns.size();
	if(n < 2) return;
	sort(fns.begin(), fns.end());
	int i = 0;
	while(1) {
		string tmp = string(path) + "/" + fns[i];
		rm_dir(tmp.c_str());
		cout << "rm " << tmp << endl;
		++i;
		if(i > n-2 || GetDirectorySize(path) < thr) break;
	}
}

int main() {
    char path[] = {"/home/lwd/code/cpp/dir"};
    string t = getCurrentTimeStr();
    t = string(path) + "/" + t.substr(0, 8);
    cout << CreateFolder((char*)t.c_str()) << endl;
    cout << GetDirectorySize(path) << endl;
    DeleteRedundancy(path, 1325490);
    if(opendir("/home/lwd/code/cpp/dir/calibration/224.jpg")==NULL) cout << "NULL\n";
    return 0;
}
```
