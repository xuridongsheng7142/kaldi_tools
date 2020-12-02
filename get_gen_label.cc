#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

//int main(){
//  map<string, string> namemap;
//  namemap["haha"] = "hehe";
//  cout << namemap["haha"] << endl;
//  return 0;
//}

void SplitString(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

string delete_string(string str, char c)
{
    string::iterator it;
    for (it = str.begin(); it < str.end(); it++)
    {
        if (*it == c)
        {
            str.erase(it);
            it--;
            /*
            it--很重要，因为使用erase()删除it指向的字符后，后面的字符就移了过来，
            it指向的位置就被后一个字符填充了，而for语句最后的it++，又使it向后移
            了一个位置，所以就忽略掉了填充过来的这个字符。在这加上it--后就和for
            语句的it++抵消了，使迭代器能够访问所有的字符。
            */
        }
    }
    return str;
}

int main(int argc,char *argv[])
{
    if(argc <= 3){
        printf("Usage: get_gen_label <utt2spk> <vad_txt> <label_one_hot>\n");
        return -1;
    }
    string utt2spk = argv[1];
    string vad_txt = argv[2];
    string label_one_hot = argv[3];
    
    // 读utt2spk文件，获取id2gen映射
    ifstream infile;
//    string file = "/Users/wangxudong/Desktop/workspace/hello/hello/hello/File";
    string file = utt2spk;
    infile.open(file.data());
    string s;
    map<string, string> namemap;
    while(getline(infile,s))
    {
//        cout << s << endl;
        vector<string> v;
        SplitString(s, v, " ");
        string id = v[0];
        string gen = v[1];
        namemap[id] = gen;
//        cout << "id: " << id << endl;
//        cout << "gen: " << gen << endl;
    }
    infile.close();             //关闭文件输入流
//    cout << namemap["20170001P00004A0022"] << endl;

    // 读vad结果文件，将结果映射到one hot形式
    vector<string> vad_value;
//    string vad_file = "/Users/wangxudong/Desktop/workspace/hello/hello/hello/vad.txt";
    string vad_file = vad_txt;
    infile.open(vad_file.data());
    string vad_s;
    string voice_flag = "1";
    string silence_flag = "0";
    string count = "";
    
    while(getline(infile, vad_s))
    {
        vector<string> vad_v;
        SplitString(vad_s, vad_v, "  ");
        string id = vad_v[0];
//        cout << vad_s << endl;
        string vad_info = vad_v[1];
        
        string target1 = "[ ";
        string target2 = " ]";
        float n1 = target1.size();
        float n2 = target2.size();
        
        vad_info = vad_info.replace(vad_info.find(target1), n1, "");
        vad_info = vad_info.replace(vad_info.find(target2), n2, "");
//        cout << id << " " << vad_info << endl;

//        cout << "man or woman: " << namemap[id] << endl;

        SplitString(vad_info, vad_value, " ");
        
        string one_count = id + " [";

        if(strcmp(namemap[id].c_str(), "M")==0){
            for(int i = 0; i != vad_value.size(); ++i){
                string vad_result = vad_value[i];
//                cout << vad_result << endl;
                if(strcmp(vad_result.c_str(), voice_flag.c_str())==0){
                    string tmp_count = "\n1.0 0.0 0.0";
                    one_count.append(tmp_count);
                }
                else if (strcmp(vad_result.c_str(), silence_flag.c_str())==0){
                    string tmp_count = "\n0.0 0.0 1.0";
                    one_count.append(tmp_count);
                }
            }
            one_count += " ]\n";
//            cout << one_count << endl;
        }
        else if (strcmp(namemap[id].c_str(), "F")==0){
            for(int i = 0; i != vad_value.size(); ++i){
                string vad_result = vad_value[i];
//                cout << vad_result << endl;
                if(strcmp(vad_result.c_str(), voice_flag.c_str())==0){
                    string tmp_count = "\n0.0 1.0 0.0";
                    one_count.append(tmp_count);
                }
                else if (strcmp(vad_result.c_str(), silence_flag.c_str())==0){
                    string tmp_count = "\n0.0 0.0 1.0";
                    one_count.append(tmp_count);
                }
            }
            one_count += " ]\n";
//            cout << one_count << endl;
        }
//        cout << one_count << endl;
        ofstream write(label_one_hot, ios::app);
        write << one_count;
//        count.append(one_count);
    }
//    write.close();
//    cout << count << endl;
//    ofstream OutFile(label_one_hot);
//    OutFile << count;
//    OutFile.close();

    infile.close();
    return 0;
}
