#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<string>
#include<time.h>
#include<random>
using namespace std;

vector<vector<float>> df,train,test,v,w;
float B=0.001,weight_change=10000,A=2,train_percent=50;
int epoch=4,cnt=0,units_in_hlayer=4;

ifstream authfile("banknote_auth.txt");
ofstream fout("output.txt");

int randomnumber() 
{
    // return 0;
    return 0 + (rand() % 2);
}

void randomweights()
{
    for(int i=0;i<v.size();i++)
        for(int j=0;j<v[i].size();j++)v[i][j]=randomnumber();
    for(int i=0;i<w.size();i++)
        for(int j=0;j<w[i].size();j++)w[i][j]=randomnumber();  
}

vector<float> calculate_zinj(vector<float> inp)
{

    vector<float> res;
    for(int j=0;j<v[0].size();j++)
    {
        float ires=v[0][j];
        for(int i=1;i<v.size();i++)
            ires+=inp[i-1]*v[i][j];
        res.push_back(ires);
    }
    return res;
}

float sigmoid(float x) {return (1.0f / (1.0f + exp(-x)));}

vector<float> activate(vector<float> inp)
{
    vector<float> res;
    for(int i=0;i<inp.size();i++)
        res.push_back(sigmoid(inp[i]));
    return res;
}

vector<float> calculate_yink(vector<float> inp)
{
    vector<float> res;
    for(int j=0;j<w[0].size();j++)
    {
        float ires=w[0][j];
        for(int i=1;i<w.size();i++)
            ires+=inp[i-1]*w[i][j];
        res.push_back(ires);
    }
    return res;
}

vector<float> calculate_delk(vector<float> inp,vector<float> target)
{
    vector<float> res;
    for(int i=0;i<inp.size();i++)
        res.push_back((target[i]-inp[i])*(inp[i]*(1-inp[i])));
    return res;
}

vector<float> calculate_delinj(vector<float> inp)
{
    vector<float> res;
    for(int i=1;i<w.size();i++)
    {
        float ires=0;
        for(int j=0;j<inp.size();j++)
            ires+=(inp[j]*w[i][j]);
        res.push_back(ires);
    }
    return res;
}

vector<float> calculate_delj(vector<float> inp1,vector<float> inp2)
{
    vector<float> res;
    for(int i=0;i<inp1.size();i++)
        res.push_back(inp1[i]*inp2[i]*(1-inp2[i]));
    return res;
}

float updateweights(vector<float>del,vector<float>inp,vector<vector<float>> &weight)
{
    float change,res=0;
    for(int j=0;j<del.size();j++)
    {
        change=A*del[j];
        weight[0][j]+=change;
        res+=abs(change);
        for(int i=0;i<inp.size();i++)
        {
            change=A*del[j]*inp[i];
            weight[i+1][j]+=change;
            res+=abs(change);
        }
    }
    return res;
}

void print_weights()
{
    fout<<"\tBack Propagation Network\t"<<endl<<endl;
    fout<<"\t\tv[i][j]\t"<<endl<<endl;
    for(int i=0;i<v.size();i++)
    {
        for(int j=0;j<v[i].size();j++)
            fout<<"\t"<<v[i][j]<<" ";
        fout<<endl;
    }
    fout<<endl<<"\t\tw[j][k]\t"<<endl<<endl;
    for(int i=0;i<w.size();i++)
    {
        for(int j=0;j<w[i].size();j++)
            fout<<"\t\t"<<w[i][j]<<" ";
        fout<<endl;
    }
}

void find_accuracy(vector<vector<float>> test)
{
    float right=0,wrong=0,trueneg=0,truepos=0,falseneg=0,falsepos=0;
    for(int i=0;i<test.size();i++)
        {
            vector<float> zinj=calculate_zinj(test[i]);
            vector<float> zj=activate(zinj);
            vector<float> yink=calculate_yink(zj);
            vector<float> yk=activate(yink);
            for(int j=0;j<yk.size();j++)
            {
                float predicted_opclass=yk[j]>=0.5?1:0;
                if(test[i][4]==predicted_opclass)
                {
                    if(predicted_opclass==0)trueneg++;
                    else truepos++;
                    right++;
                }
                else 
                {
                    if(predicted_opclass==0)falseneg++;
                    else falsepos++;
                    wrong++;
                }
            }
        }
    fout<<"\t\tconfusion matrix"<<endl;
    fout<<"\ttruePositive"<<" trueNegative"<<endl;
    fout<<"\t"<<truepos<<"\t\t\t "<<trueneg<<endl;
    fout<<"\tfalsePositive"<<" falseNegative"<<endl;
    fout<<"\t"<<falsepos<<"\t\t\t "<<falseneg<<endl<<endl;
    float sum=truepos+trueneg+falsepos+falseneg;
    float Precision = truepos / (trueneg + falsepos);
    float Recall = truepos / (truepos + falseneg);
    float FMeasure = (2 * Precision * Recall) / (Precision + Recall);
    fout<<"\taccuracy="<<" "<<(truepos+trueneg)*100/sum<<"%"<<endl;
    fout<<"\tprecision="<<Precision<<endl;
    fout<<"\trecall="<<Recall<<endl;
    fout<<"\tfscore="<<FMeasure<<endl;
}
    

int main()
{
    float variance,skewness,kurtosis,entropy,opclass;
    string data;
    
    while(!authfile.eof())
    {
        getline(authfile, data,',');variance=stof(data);
        getline(authfile, data,',');skewness=stof(data);
        getline(authfile, data,',');kurtosis=stof(data);
        getline(authfile, data,',');entropy=stof(data);
        getline(authfile, data);opclass=stoi(data);
        df.push_back({variance,skewness,kurtosis,entropy,opclass});
        // cout <<variance<<" "<<skewness<<" "<<kurtosis<<" "<<entropy<<" "<<opclass<<" "<< endl;
    }

    srand(time(0));
    shuffle(df.begin(),df.end(),default_random_engine(time(0)));// can call without parameter time(0)
    shuffle(df.begin(),df.end(),default_random_engine(time(0)));
    int sz=df.size();int trainsz=train_percent*df.size()/100;
    train=vector<vector<float>>(df.begin(),df.begin()+trainsz);
    test=vector<vector<float>>(df.begin()+trainsz,df.end());    
    
    v.assign(5,vector<float>(units_in_hlayer,0));
    w.assign(units_in_hlayer+1,vector<float>(1,0));
    randomweights();
    cout<<endl;
    while(weight_change>B && cnt<epoch)
    {
        weight_change=0;
        for(int i=0;i<train.size();i++)
        {
            // feed-forward phase
            vector<float> zinj=calculate_zinj(train[i]);
            vector<float> zj=activate(zinj);
            vector<float> yink=calculate_yink(zj);
            vector<float> yk=activate(yink);
            // back-propagation phase
            vector<float> delk=calculate_delk(yk,{train[i][4]});
            vector<float> delinj=calculate_delinj(delk);
            vector<float> delj=calculate_delj(delinj,zj);
            // weight-updation
            weight_change+=updateweights(delk,zj,w)+updateweights(delj,{train[i][0],train[i][1],train[i][2],train[i][3]},v);
        }
        cnt++;
        A=(A-(A/5));
        // A=A/2;
        cout<<"weightChange="<<weight_change<<"\t\tepoch="<<cnt<<"\t\tlearning rate="<<A<<endl;
    }

    print_weights();
    fout<<endl<<"train size="<<train.size()<<" | test size="<<test.size()<<endl<<endl;
    fout<<"Train Phase : "<<endl;
    find_accuracy(train);
    fout<<endl;
    fout<<"Test Phase : "<<endl;
    find_accuracy(test);
    authfile.close();
    fout.close();
    return 0;
}






















// float updateweightW(vector<float>delk,vector<float>zj)
// {
//     float change,res=0;
//     for(int j=0;j<delk.size();j++)
//     {
//         change=A*delk[j];
//         w[0][j]+=change;
//         res+=abs(change);
//         for(int i=0;i<zj.size();i++)
//         {
//             change=A*delk[j]*zj[i];
//             w[i+1][j]+=change;
//             res+=abs(change);
//         }
//     }
//     return res;
// }


// float updateweightV(vector<float>delj,vector<float>inp)
// {
//     float change,res=0;
//     for(int j=0;j<delj.size();j++)
//     {
//         change=A*delj[j];
//         v[0][j]+=change;
//         res+=abs(change);
//         for(int i=0;i<inp.size();i++)
//         {
//             change=A*delj[j]*inp[i];
//             v[i+1][j]+=change;
//             res+=abs(change);
//         }
//     }
//     return res;
// }