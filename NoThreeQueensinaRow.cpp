#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <climits>
#include <utility>
#include <string>
#include <iomanip>

using namespace std;

// --- Timing helpers ---
static inline double seconds_since(const chrono::steady_clock::time_point& t){
    return chrono::duration<double>(chrono::steady_clock::now() - t).count();
}
static inline long long ms_since(const chrono::steady_clock::time_point& t){
    return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t).count();
}

struct SwapEval{ int j; long long dline; long long ddiag; long long obj; };

struct Opts{int N=999; long long budget_mul=1500; int sample_base=256; int sample_cap=1024; int max_restart=32; int seconds=25; unsigned threads=0; unsigned seed=0; bool verbose=false; bool dump_best=false; int log_ms=0; int wdiag=64; bool hard_diag=false; bool exact=false; bool print_time=false; bool compose=false; int comp_a=0; int comp_b=0;};
static inline bool isnum(const string& s){ if(s.empty()) return false; for(char c:s) if(c<'0'||c>'9') return false; return true; }
static Opts parse_opts(int argc,char** argv){
    Opts o;
    for(int i=1;i<argc;++i){
        string a=argv[i];
        auto read_next = [&](int& i)->string{
            if(i+1<argc) return string(argv[++i]);
            return string();
        };
        if(isnum(a)) o.N=stoi(a);
        else if(a=="--time"){ string v=read_next(i); if(!v.empty()) o.seconds=stoi(v); }
        else if(a=="--budget"){ string v=read_next(i); if(!v.empty()) o.budget_mul=stoll(v); }
        else if(a=="--sample"){ string v=read_next(i); if(!v.empty()) o.sample_base=stoi(v); }
        else if(a=="--cap"){ string v=read_next(i); if(!v.empty()) o.sample_cap=stoi(v); }
        else if(a=="--restarts"){ string v=read_next(i); if(!v.empty()) o.max_restart=stoi(v); }
        else if(a=="--threads"){ string v=read_next(i); if(!v.empty()) o.threads=stoul(v); }
        else if(a=="--seed"){ string v=read_next(i); if(!v.empty()) o.seed=stoul(v); }
        else if(a=="--wdiag"){ string v=read_next(i); if(!v.empty()) o.wdiag=stoi(v); }
        else if(a.rfind("--time=",0)==0) o.seconds=stoi(a.substr(7));
        else if(a.rfind("--budget=",0)==0) o.budget_mul=stoll(a.substr(9));
        else if(a.rfind("--sample=",0)==0) o.sample_base=stoi(a.substr(9));
        else if(a.rfind("--cap=",0)==0) o.sample_cap=stoi(a.substr(6));
        else if(a.rfind("--restarts=",0)==0) o.max_restart=stoi(a.substr(11));
        else if(a.rfind("--threads=",0)==0) o.threads=stoul(a.substr(10));
        else if(a.rfind("--seed=",0)==0) o.seed=stoul(a.substr(7));
        else if(a=="--verbose") o.verbose=true;
        else if(a=="--dump-best") o.dump_best=true;
        else if(a.rfind("--log-ms=",0)==0) o.log_ms=stoi(a.substr(9));
        else if(a.rfind("--wdiag=",0)==0) o.wdiag=stoi(a.substr(8));
        else if(a=="--hard-diag") o.hard_diag=true;
        else if(a=="--exact") o.exact=true;
        else if(a=="--print-time") o.print_time=true;
        else if(a.rfind("--compose=",0)==0){
            string s=a.substr(10);
            char sep=0;
            if(s.find('x')!=string::npos) sep='x';
            else if(s.find('X')!=string::npos) sep='X';
            else if(s.find('*')!=string::npos) sep='*';
            if(sep){
                size_t p=s.find(sep);
                o.comp_a=stoi(s.substr(0,p));
                o.comp_b=stoi(s.substr(p+1));
                o.compose=true;
            }
        }
    }
    return o;
}
// --- Composition helper ---
static vector<int> compose_product(const vector<int>& A, const vector<int>& B){
    int n0=(int)A.size(), s=(int)B.size();
    vector<int> P(n0*s);
    for(int r=0;r<n0;++r){
        for(int t=0;t<s;++t){
            P[r*s + t] = s*A[r] + B[t];
        }
    }
    return P;
}

static vector<vector<int>> GG;
static inline int g(int a,int b){a=abs(a);b=abs(b);while(b){int t=a%b;a=b;b=t;}return a;}
static inline long long keynorm(int dx,int dy){
    int adx=dx<0?-dx:dx, ady=dy<0?-dy:dy; int gg=GG[adx][ady]; if(gg==0) gg=1;
    dx/=gg; dy/=gg; if(dx<0 || (dx==0 && dy<0)){ dx=-dx; dy=-dy; }
    return ((long long)dx<<32) ^ (unsigned long long)(dy & 0xffffffffu);
}

// --- Line geometry helpers ---
struct LineKey{
    int dx, dy;
    long long k;
    bool operator==(const LineKey& o) const noexcept { return dx==o.dx && dy==o.dy && k==o.k; }
};
struct LineKeyHash{
    size_t operator()(const LineKey& L) const noexcept{
        uint64_t a=(uint32_t)L.dx, b=(uint32_t)L.dy; uint64_t c=(uint64_t)L.k;
        uint64_t h=a*0x9e3779b185ebca87ULL ^ (b+0x9e3779b185ebca87ULL + (a<<6) + (a>>2));
        h ^= c + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        return (size_t)h;
    }
};
static inline LineKey make_line_key(int x1,int y1,int x2,int y2){
    int dx=x2-x1, dy=y2-y1;
    if(dx==0 && dy==0){ return {0,0,0}; }
    int s= (dx<0 || (dx==0 && dy<0))? -1: 1;
    dx*=s; dy*=s;
    int gg=g(abs(dx),abs(dy)); if(gg>1){ dx/=gg; dy/=gg; }
    long long k=(long long)dx*y1 - (long long)dy*x1;
    return {dx,dy,k};
}

// --- Exact backtracking solver ---
static bool exact_dfs(int r,
                      int N,
                      vector<int>& col,
                      vector<char>& used,
                      vector<char>& d1,
                      vector<char>& d2,
                      unordered_set<LineKey,LineKeyHash>& lines){
    if(r==N) return true;
    for(int c=0;c<N;++c){
        if(used[c]) continue;
        int a=r-c+N, b=r+c;
        if(d1[a] || d2[b]) continue;
        bool ok=true;
        vector<LineKey> added;
        added.reserve(r);
        for(int pr=0; pr<r; ++pr){
            int pc=col[pr];
            LineKey L=make_line_key(r,c,pr,pc);
            if(lines.find(L)!=lines.end()){ ok=false; break; }
            added.push_back(L);
        }
        if(!ok) continue;
        col[r]=c; used[c]=1; d1[a]=1; d2[b]=1;
        for(auto& L: added) lines.insert(L);
        if(exact_dfs(r+1,N,col,used,d1,d2,lines)) return true;
        for(auto& L: added) lines.erase(L);
        used[c]=0; d1[a]=0; d2[b]=0; col[r]=-1;
    }
    return false;
}

static bool exact_solve_perm_no3(int N, vector<int>& out){
    vector<int> col(N,-1);
    vector<char> used(N,0), d1(2*N+1,0), d2(2*N+1,0);
    unordered_set<LineKey,LineKeyHash> lines;
    bool ok=exact_dfs(0,N,col,used,d1,d2,lines);
    if(ok){ out=col; }
    return ok;
}
static inline unsigned hw_threads(){ unsigned t=thread::hardware_concurrency(); if(!t) t=8; return min<unsigned>(t,64); }
static inline void init_gcd_lut(int N){
    GG.assign(N+1, vector<int>(N+1,0));
    for(int i=0;i<=N;++i) for(int j=0;j<=N;++j) GG[i][j]=g(i,j);
}
static vector<int> nqueen_seed(int N){
    vector<int> even,odd,perm,col;
    for(int v=2; v<=N; v+=2) even.push_back(v);
    for(int v=1; v<=N; v+=2) odd.push_back(v);
    if(N%6==2){
        if((int)odd.size()>=2) swap(odd[0],odd[1]);
        if((int)odd.size()>=3){ int v=odd[2]; odd.erase(odd.begin()+2); odd.push_back(v); }
    } else if(N%6==3){
        if(!even.empty()){ int v=even[0]; even.erase(even.begin()); even.push_back(v); }
        if((int)odd.size()>=2){ int a=odd[0], b=odd[1]; odd.erase(odd.begin(),odd.begin()+2); odd.push_back(a); odd.push_back(b); }
    }
    perm=even; perm.insert(perm.end(),odd.begin(),odd.end());
    col.resize(N);
    for(int i=0;i<N;++i) col[i]=perm[i]-1;
    return col;
}
static void build_scnt_rconf_parallel(const vector<int>& col, vector<unordered_map<long long,int>>& scnt, vector<long long>& rconf, unsigned threads){
    int n=(int)col.size(); scnt.assign(n,{}); rconf.assign(n,0);
    int chunk=(n+threads-1)/threads;
    vector<thread> th;
    for(unsigned t=0;t<threads;++t){
        int L=t*chunk, R=min(n,(int)((t+1)*chunk));
        th.emplace_back([&,L,R](){
            for(int i=L;i<R;++i){
                auto& mp=scnt[i]; mp.reserve(n*2);
                long long s=0;
                for(int j=0;j<n;++j){
                    if(j==i) continue;
                    int dx=j-i, dy=col[j]-col[i];
                    long long k=keynorm(dx,dy);
                    int &c=mp[k];
                    if(++c==2) s+=1; else if(c>2) s+=c-1;
                }
                rconf[i]=s;
            }
        });
    }
    for(auto& t:th) t.join();
}
static long long row_conf_if(int i,int newCol,const vector<int>& col){
    int n=(int)col.size();
    unordered_map<long long,int> mp; mp.reserve(n*2);
    long long s=0;
    for(int t=0;t<n;++t){
        if(t==i) continue;
        int dx=t-i, dy=col[t]-newCol;
        long long k=keynorm(dx,dy);
        int &c=mp[k];
        if(++c==2) s+=1; else if(c>2) s+=c-1;
    }
    return s;
}
static inline void dec_count(unordered_map<long long,int>& mp,long long key,long long& rsum){ auto it=mp.find(key); if(it!=mp.end()){ rsum -= (it->second - 1); if(--(it->second)==0) mp.erase(it);} }
static inline void inc_count(unordered_map<long long,int>& mp,long long key,long long& rsum){ int &c=mp[key]; rsum += c; ++c; }
static inline long long pairs2(long long c);
static long long delta_if_swap(int i,int j,const vector<int>& col,const vector<unordered_map<long long,int>>& scnt,const vector<long long>& rconf){
    int n=(int)col.size();
    long long d=0;
    long long newI=row_conf_if(i,col[j],col), newJ=row_conf_if(j,col[i],col);
    d += (newI - rconf[i]) + (newJ - rconf[j]);
    for(int t=0;t<n;++t){
        if(t==i||t==j) continue;
        long long k1=keynorm(i-t,col[i]-col[t]);
        long long k2=keynorm(i-t,col[j]-col[t]);
        long long k3=keynorm(j-t,col[j]-col[t]);
        long long k4=keynorm(j-t,col[i]-col[t]);
        long long keys[4]={k1,k2,k3,k4};
        int deltas[4]={-1,+1,-1,+1};
        long long ukey[4]; int udelta[4]; int ucnt=0;
        for(int m=0;m<4;++m){
            long long kk=keys[m]; int dd=deltas[m];
            int pos=-1;
            for(int p=0;p<ucnt;++p){ if(ukey[p]==kk){ pos=p; break; } }
            if(pos==-1){ ukey[ucnt]=kk; udelta[ucnt]=dd; ++ucnt; }
            else { udelta[pos]+=dd; }
        }
        long long dsum=0;
        for(int p=0;p<ucnt;++p){
            int dd=udelta[p];
            if(dd==0) continue;
            auto it=scnt[t].find(ukey[p]);
            long long c=(it==scnt[t].end()?0:it->second);
            dsum += pairs2(c + dd) - pairs2(c);
        }
        d += dsum;
    }
    return d;
}
static inline int diag_after_swap(int i,int j,const vector<int>& col,const vector<int>& d1,const vector<int>& d2,int off){
    int ci=col[i],cj=col[j];
    int i_d1_old=i-ci+off,i_d2_old=i+ci,j_d1_old=j-cj+off,j_d2_old=j+cj;
    int i_d1_new=i-cj+off,i_d2_new=i+cj,j_d1_new=j-ci+off,j_d2_new=j+ci;
    int s=0;
    {int c=d1[i_d1_new]; if(i_d1_new==j_d1_old) c--; if(i_d1_new==i_d1_old) c--; s+=c;}
    {int c=d2[i_d2_new]; if(i_d2_new==j_d2_old) c--; if(i_d2_new==i_d2_old) c--; s+=c;}
    {int c=d1[j_d1_new]; if(j_d1_new==i_d1_old) c--; if(j_d1_new==j_d1_old) c--; s+=c;}
    {int c=d2[j_d2_new]; if(j_d2_new==i_d2_old) c--; if(j_d2_new==j_d2_old) c--; s+=c;}
    return s;
}
static inline long long pairs2(long long c){ return c*(c-1)/2; }
static inline long long diag_pairs_delta_swap(int i,int j,const vector<int>& col,const vector<int>& d1,const vector<int>& d2,int off){
    int ci=col[i], cj=col[j];
    int keys_t[8]; int typ[8]; int delta[8]; int len=0;
    auto push = [&](int t, int idx, int val){
        for(int m=0;m<len;++m){ if(typ[m]==t && keys_t[m]==idx){ delta[m]+=val; return; } }
        typ[len]=t; keys_t[len]=idx; delta[len]=val; ++len;
    };
    push(0, i-ci+off, -1);
    push(1, i+ci,      -1);
    push(0, j-cj+off,  -1);
    push(1, j+cj,      -1);
    push(0, i-cj+off,  +1);
    push(1, i+cj,      +1);
    push(0, j-ci+off,  +1);
    push(1, j+ci,      +1);
    long long dsum=0;
    for(int m=0;m<len;++m){
        int t=typ[m], idx=keys_t[m], dv=delta[m];
        int c = (t==0? d1[idx] : d2[idx]);
        dsum += pairs2(c + dv) - pairs2(c);
    }
    return dsum;
}
static void apply_swap_update(int i,int j, vector<int>& col, vector<unordered_map<long long,int>>& scnt, vector<long long>& rconf){
    int n=(int)col.size(); int ci=col[i], cj=col[j];
    for(int t=0;t<n;++t){ if(t==i||t==j) continue;
        long long k_i_old=keynorm(t-i, col[t]-ci); dec_count(scnt[i],k_i_old,rconf[i]);
        long long k_i_new=keynorm(t-i, col[t]-cj); inc_count(scnt[i],k_i_new,rconf[i]);
        long long k_t_old=keynorm(i-t, ci-col[t]); dec_count(scnt[t],k_t_old,rconf[t]);
        long long k_t_new=keynorm(i-t, cj-col[t]); inc_count(scnt[t],k_t_new,rconf[t]);
        long long k_j_old=keynorm(t-j, col[t]-cj); dec_count(scnt[j],k_j_old,rconf[j]);
        long long k_j_new=keynorm(t-j, col[t]-ci); inc_count(scnt[j],k_j_new,rconf[j]);
        long long k_tj_old=keynorm(j-t, cj-col[t]); dec_count(scnt[t],k_tj_old,rconf[t]);
        long long k_tj_new=keynorm(j-t, ci-col[t]); inc_count(scnt[t],k_tj_new,rconf[t]);
    }
    long long k_ij_old=keynorm(j-i, cj-ci); dec_count(scnt[i],k_ij_old,rconf[i]);
    long long k_ij_new=keynorm(j-i, ci-cj); inc_count(scnt[i],k_ij_new,rconf[i]);
    long long k_ji_old=keynorm(i-j, ci-cj); dec_count(scnt[j],k_ji_old,rconf[j]);
    long long k_ji_new=keynorm(i-j, cj-ci); inc_count(scnt[j],k_ji_new,rconf[j]);
    col[i]=cj; col[j]=ci;
}
static inline long long sum_rconf(const vector<long long>& rconf){ long long s=0; for(long long v: rconf) s+=v; return s; }
static inline void do_swap_update(int i,int j, vector<int>& col, vector<int>& d1, vector<int>& d2, int off, vector<unordered_map<long long,int>>& scnt, vector<long long>& rconf){
    int i_d1=i-col[i]+off,i_d2=i+col[i],j_d1=j-col[j]+off,j_d2=j+col[j];
    d1[i_d1]--; d2[i_d2]--; d1[j_d1]--; d2[j_d2]--;
    apply_swap_update(i,j,col,scnt,rconf);
    d1[i-col[i]+off]++; d2[i+col[i]]++; d1[j-col[j]+off]++; d2[j+col[j]]++;
}
static int worst_partner(int i,const vector<int>& col,const vector<unordered_map<long long,int>>& scnt){
    long long bestc=0; long long key=0; bool has=false;
    for(auto& kv:scnt[i]) if(kv.second>bestc){bestc=kv.second; key=kv.first; has=true;}
    if(!has||bestc<=1) return -1;
    int n=(int)col.size();
    for(int t=0;t<n;++t){ if(t==i) continue; int dx=t-i, dy=col[t]-col[i]; if(keynorm(dx,dy)==key) return t; }
    return -1;
}
static SwapEval pick_best_swap_parallel(int i,
    const vector<int>& col,
    const vector<unordered_map<long long,int>>& scnt,
    const vector<long long>& rconf,
    const vector<int>& d1,
    const vector<int>& d2,
    int off,
    mt19937& rng,
    unsigned T,
    int sample,
    int wdiag,
    bool hard_diag){
    int n=(int)col.size();
    long long keyW=0, cntW=0;
    for(const auto& kv: scnt[i]) if(kv.second>cntW){ cntW=kv.second; keyW=kv.first; }
    vector<int> cand;
    cand.reserve(sample + (int)cntW + 8);
    // Enhanced candidate generation: gather top-3 slope keys (by count > 1) and add their members
    vector<pair<long long,int>> top;
    top.reserve(scnt[i].size());
    for(const auto& kv: scnt[i]) if(kv.second>1) top.push_back({kv.first, kv.second});
    sort(top.begin(), top.end(), [](const pair<long long,int>& a, const pair<long long,int>& b){ return a.second>b.second; });
    int K = min(3, (int)top.size());
    vector<char> usedRow(n, 0);
    usedRow[i]=1;
    auto add_row = [&](int t){
        if(!usedRow[t]){ usedRow[t]=1; cand.push_back(t); }
    };
    for(int kk=0; kk<K; ++kk){
        long long key = top[kk].first;
        for(int t=0;t<n;++t){
            if(t==i) continue;
            if(keynorm(t-i, col[t]-col[i])==key) add_row(t);
        }
    }
    // Deduplicating candidate block and support for full scan if sample >= n-1
    if(sample >= n-1){
        cand.clear();
        cand.reserve(n-1);
        for(int t=0;t<n;++t) if(t!=i) cand.push_back(t);
    } else {
        if((int)cand.size()<sample){
            while((int)cand.size()<sample){
                int x=rng()%n;
                if(!usedRow[x]){ usedRow[x]=1; cand.push_back(x); }
                if((int)cand.size()>=n-1) break;
            }
        }
    }
    if(cand.empty()) return {-1,0,0,0};
    atomic<int> nxt(0);
    struct BestLoc{ long long obj; int j; long long dline; long long ddiag; };
    vector<BestLoc> best(T); for(unsigned t=0;t<T;++t){ best[t].obj=LLONG_MAX; best[t].j=-1; best[t].dline=0; best[t].ddiag=0; }
    vector<thread> th; th.reserve(T);
    for(unsigned t=0;t<T;++t){
        th.emplace_back([&,t](){
            BestLoc loc; loc.obj=LLONG_MAX; loc.j=-1; loc.dline=0; loc.ddiag=0;
            int k;
            while((k=nxt.fetch_add(1, memory_order_relaxed)) < (int)cand.size()){
                int j=cand[k];
                if(j==i) continue;
                long long ddiag = 0;
                if(hard_diag){
                    if(diag_after_swap(i,j,col,d1,d2,off)!=0) continue;
                } else {
                    ddiag = diag_pairs_delta_swap(i,j,col,d1,d2,off);
                }
                long long dline = delta_if_swap(i,j,col,scnt,rconf);
                long long obj = dline + (long long)wdiag * ddiag;
                if(obj<loc.obj){
                    loc.obj=obj; loc.j=j; loc.dline=dline; loc.ddiag=ddiag;
                }
            }
            best[t]=loc;
        });
    }
    for(auto& t:th) t.join();
    BestLoc ans=best[0];
    for(const auto& b: best) if(b.obj<ans.obj) ans=b;
    if(ans.j==-1) return {-1,0,0,0};
    return {ans.j, ans.dline, ans.ddiag, ans.obj};
}
static bool diag_reduce_phase(vector<int>& col,
                              vector<int>& d1,
                              vector<int>& d2,
                              int off,
                              vector<unordered_map<long long,int>>& scnt,
                              vector<long long>& rconf,
                              long long& dpp,
                              long long& row_total,
                              long long& obj_total,
                              mt19937& rng,
                              unsigned T,
                              const Opts& O){
    int N=(int)col.size();
    int limit = N*16;
    for(int it=0; it<limit && dpp>0; ++it){
        int i = rng()%N;
        auto res = pick_best_swap_parallel(i,col,scnt,rconf,d1,d2,off,rng,T,min(512, max(64, O.sample_base)), 1, true);
        if(res.j!=-1 && res.ddiag<0){
            int j=res.j;
            int i_d1=i-col[i]+off,i_d2=i+col[i],j_d1=j-col[j]+off,j_d2=j+col[j];
            d1[i_d1]--; d2[i_d2]--; d1[j_d1]--; d2[j_d2]--;
            apply_swap_update(i,j,col,scnt,rconf);
            d1[i-col[i]+off]++; d2[i+col[i]]++; d1[j-col[j]+off]++; d2[j+col[j]]++;
            dpp += res.ddiag;
            long long nt=0; for(long long v:rconf) nt+=v; row_total=nt;
            obj_total = row_total + dpp;
        }
    }
    return dpp==0;
}
static bool three_cycle_escape(vector<int>& col,
                               vector<int>& d1,
                               vector<int>& d2,
                               int off,
                               vector<unordered_map<long long,int>>& scnt,
                               vector<long long>& rconf,
                               long long& dpp,
                               long long& row_total,
                               long long& obj_total,
                               mt19937& rng,
                               unsigned T,
                               const Opts& O){
    int N=(int)col.size();
    int i=0; long long worst=-1; for(int r=0;r<N;++r) if(rconf[r]>worst){ worst=rconf[r]; i=r; }
    if(worst<=0) return false;
    int trials = min(64, N-2);
    for(int t=0; t<trials; ++t){
        int j=rng()%N; if(j==i){ --t; continue; }
        int k=rng()%N; if(k==i || k==j){ --t; continue; }
        do_swap_update(i,j,col,d1,d2,off,scnt,rconf);
        do_swap_update(j,k,col,d1,d2,off,scnt,rconf);
        long long new_row = sum_rconf(rconf);
        long long new_dpp = 0; for(int c: d1) if(c>=2) new_dpp += pairs2(c); for(int c: d2) if(c>=2) new_dpp += pairs2(c);
        long long new_obj = new_row + (long long)O.wdiag * new_dpp;
        if(new_obj < obj_total){ row_total=new_row; dpp=new_dpp; obj_total=new_obj; return true; }
        do_swap_update(j,k,col,d1,d2,off,scnt,rconf);
        do_swap_update(i,j,col,d1,d2,off,scnt,rconf);
        do_swap_update(i,k,col,d1,d2,off,scnt,rconf);
        do_swap_update(i,j,col,d1,d2,off,scnt,rconf);
        new_row = sum_rconf(rconf);
        new_dpp = 0; for(int c: d1) if(c>=2) new_dpp += pairs2(c); for(int c: d2) if(c>=2) new_dpp += pairs2(c);
        new_obj = new_row + (long long)O.wdiag * new_dpp;
        if(new_obj < obj_total){ row_total=new_row; dpp=new_dpp; obj_total=new_obj; return true; }
        do_swap_update(i,j,col,d1,d2,off,scnt,rconf);
        do_swap_update(i,k,col,d1,d2,off,scnt,rconf);
    }
    return false;
}
static bool check_parallel(const vector<int>& col){
    int n=(int)col.size();
    vector<int> seen(n,0);
    for(int i=0;i<n;++i){ if(col[i]<0||col[i]>=n) return false; if(++seen[col[i]]>1) return false; }
    int off=n;
    vector<int> d1(2*n+1,0), d2(2*n+1,0);
    for(int i=0;i<n;++i){
        int a=i-col[i]+off, b=i+col[i];
        if(++d1[a]>1 || ++d2[b]>1) return false;
    }
    atomic<bool> bad{false};
    unsigned T=hw_threads();
    int chunk=(n+T-1)/T;
    vector<thread> th;
    for(unsigned t=0;t<T;++t){
        int L=t*chunk, R=min(n,(int)((t+1)*chunk));
        th.emplace_back([&,L,R](){
            unordered_map<long long,int> mp; mp.reserve(n*2);
            for(int i=L;i<R;++i){
                mp.clear();
                for(int j=0;j<n;++j){
                    if(j==i) continue;
                    int dx=j-i, dy=col[j]-col[i];
                    long long k=keynorm(dx,dy);
                    int &c=mp[k];
                    if(++c>=2){ bad=true; return; }
                }
            }
        });
    }
    for(auto& t:th) t.join();
    return !bad.load();
}
int main(int argc,char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    auto O=parse_opts(argc,argv);
    int N=O.N;
    bool start_with_col=false;
    vector<int> start_col;
    mt19937 rng(O.seed? O.seed : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    init_gcd_lut(N);
    unsigned T = O.threads? O.threads : hw_threads();
    const int MAX_RESTART=O.max_restart;
    if(O.verbose){
        cerr<<"start N="<<N<<" threads="<<T<<" budget_per_attempt="<<(long long)N*O.budget_mul<<" restarts="<<MAX_RESTART<<" time_s="<<O.seconds<<" [compose supported]\n";
    }
    // --- Fast path: composition ---
    if(O.compose){
        int n0 = O.comp_a, s = O.comp_b;
        if(n0<=0 || s<=0){ cerr<<"invalid --compose sizes\n"; return 2; }
        auto tcomp0 = chrono::steady_clock::now();
        vector<int> A,B;
        init_gcd_lut(max(n0,s));
        if(O.verbose) cerr<<"compose: solving exact for "<<n0<<" and "<<s<<"\n";
        if(!exact_solve_perm_no3(n0,A)){ cerr<<"exact failed for n0="<<n0<<"\n"; return 3; }
        if(!exact_solve_perm_no3(s,B)){ cerr<<"exact failed for s="<<s<<"\n"; return 4; }
        int NN = n0*s;
        init_gcd_lut(NN);
        auto P = compose_product(A,B);
        if(!check_parallel(P)){
            if(O.verbose) cerr<<"composed solution failed checker; attempting heuristic repair\n";
            N = NN;
            init_gcd_lut(N);
            start_with_col=true;
            start_col = P;
            if(O.verbose) cerr<<"compose-repair: effective N="<<N<<"\n";
        } else {
            if(O.verbose){
                auto t1=chrono::steady_clock::now();
                cerr<<"compose-solved N="<<n0*s<<" time_s="<<chrono::duration<double>(t1-tcomp0).count()<<"\n";
            }
            cout<<n0*s<<"\n";
            for(int i=0;i<n0*s;++i){ if(i) cout<<' '; cout<<P[i]+1; }
            cout<<"\n";
            if(O.print_time){
                auto t1=chrono::steady_clock::now();
                cout<<"elapsed_s="<<fixed<<setprecision(6)<<chrono::duration<double>(t1-tcomp0).count()<<"\n";
            }
            return 0;
        }
    }
    // --- Early exact branch ---
    if(O.exact || N<=32){
        auto te0 = chrono::steady_clock::now();
        vector<int> sol;
        if(exact_solve_perm_no3(N, sol)){
            if(O.verbose){
                auto te1 = chrono::steady_clock::now();
                cerr << "exact-solved time_s=" << chrono::duration<double>(te1 - te0).count() << "\n";
            }
            cout<<N<<"\n";
            for(int i=0;i<N;++i){ if(i) cout<<' '; cout<<sol[i]+1; }
            cout<<"\n";
            if(O.print_time){
                auto te1 = chrono::steady_clock::now();
                cout << "elapsed_s=" << fixed << setprecision(6)
                     << chrono::duration<double>(te1 - te0).count() << "\n";
            }
            return 0;
        } else if(O.verbose){
            cerr<<"exact-failed, falling back to heuristic\n";
        }
    }
    long long global_best=LLONG_MAX;
    vector<int> global_best_col;
    auto update_global_best = [&](long long tot, const vector<int>& c){
        long long safe = tot<0 ? 0 : tot;
        if(safe<global_best){ global_best=safe; global_best_col=c; }
    };
    auto t0=chrono::steady_clock::now();
    auto deadline=t0 + chrono::seconds(O.seconds);
    auto diag_pairs_total = [&](const vector<int>& D1,const vector<int>& D2){
        long long s=0;
        for(int c: D1) if(c>=2) s += 1LL*c*(c-1)/2;
        for(int c: D2) if(c>=2) s += 1LL*c*(c-1)/2;
        return s;
    };
    for(int attempt=0;attempt<MAX_RESTART;++attempt){
        auto tatt = chrono::steady_clock::now();
        long long att_iters = 0;
        vector<int> col;
        if(start_with_col && attempt==0){ col = start_col; }
        else { col = nqueen_seed(N); }
        int off=N;
        vector<int> d1(2*N+1,0), d2(2*N+1,0);
        for(int i=0;i<N;++i){ d1[i-col[i]+off]++; d2[i+col[i]]++; }
        long long dpp = diag_pairs_total(d1,d2);
        vector<unordered_map<long long,int>> scnt; vector<long long> rconf; build_scnt_rconf_parallel(col,scnt,rconf,T);
        long long row_total=0; for(long long v:rconf) row_total+=v;
        long long obj_total = row_total + (long long)O.wdiag * dpp;
        update_global_best(obj_total,col);
        long long budget=(long long)N*O.budget_mul;
        static long long last_total_snapshot=-1;
        static int stale=0;
        for(long long it=0; obj_total>0 && it<budget; ++it){
            ++att_iters;
            static auto last_log=chrono::steady_clock::now();
            if(O.log_ms>0){
                auto now=chrono::steady_clock::now();
                if(chrono::duration_cast<chrono::milliseconds>(now - last_log).count()>=O.log_ms){
                    if(O.verbose) cerr<<"att="<<attempt<<" it="<<it<<" obj="<<obj_total<<" dpp="<<dpp<<" thr="<<T<<"\n";
                    last_log=now;
                }
            }
            if((it & 511)==0){
                if(chrono::steady_clock::now()>deadline) break;
                if(O.verbose){ cerr<<"att="<<attempt<<" it="<<it<<" obj="<<obj_total<<" dpp="<<dpp<<" thr="<<T<<"\r"<<flush; }
            }
            if((it & 2047)==0){
                three_cycle_escape(col,d1,d2,off,scnt,rconf,dpp,row_total,obj_total,rng,T,O);
            }
            int i=0; long long worst=-1; for(int r=0;r<N;++r){ if(rconf[r]>worst){worst=rconf[r]; i=r;} }
            if(worst<=0){
                if(!O.hard_diag && dpp>0){ diag_reduce_phase(col,d1,d2,off,scnt,rconf,dpp,row_total,obj_total,rng,T,O); }
                break;
            }
            auto pick = [&](int sample)->SwapEval{
                auto res = pick_best_swap_parallel(i,col,scnt,rconf,d1,d2,off,rng,T,sample,O.wdiag,O.hard_diag);
                if(res.j==-1){
                    for(int tr=0; tr<2048; ++tr){
                        int jj=rng()%N; if(jj==i) continue;
                        long long ddiag = 0;
                        if(O.hard_diag){
                            if(diag_after_swap(i,jj,col,d1,d2,off)!=0) continue;
                        } else {
                            ddiag = diag_pairs_delta_swap(i,jj,col,d1,d2,off);
                        }
                        long long dline = delta_if_swap(i,jj,col,scnt,rconf);
                        long long obj = dline + (long long)O.wdiag * ddiag;
                        res = {jj, dline, ddiag, obj};
                        break;
                    }
                }
                return res;
            };
            auto best=pick(min(min(O.sample_cap,N-1), O.sample_base + (int)worst));
            int bestJ=best.j; long long dline=best.dline; long long ddiag=best.ddiag; long long bestObj=best.obj;
            if(bestJ!=-1){
                int i_d1=i-col[i]+off,i_d2=i+col[i],j_d1=bestJ-col[bestJ]+off,j_d2=bestJ+col[bestJ];
                d1[i_d1]--; d2[i_d2]--; d1[j_d1]--; d2[j_d2]--;
                apply_swap_update(i,bestJ,col,scnt,rconf);
                d1[i-col[i]+off]++; d2[i+col[i]]++; d1[bestJ-col[bestJ]+off]++; d2[bestJ+col[bestJ]]++;
                dpp += ddiag;
                long long nt=0; for(long long v:rconf) nt+=v; row_total=nt;
                obj_total = row_total + (long long)O.wdiag * dpp;
                update_global_best(obj_total,col);
            }
            if(last_total_snapshot==-1) last_total_snapshot=obj_total;
            if(bestObj>=0 || bestJ==-1) ++stale; else stale=0;
            if((it & 1023)==0){
                if(obj_total>=last_total_snapshot) ++stale; else stale=0;
                last_total_snapshot=obj_total;
            }
            if(stale> (int)N/2){
                auto shake=[&](){
                    for(int tr=0; tr<2000; ++tr){
                        int a=rng()%N, b=rng()%N; if(a==b) continue;
                        long long ddiag = O.hard_diag ? 0 : diag_pairs_delta_swap(a,b,col,d1,d2,off);
                        if(O.hard_diag ? (diag_after_swap(a,b,col,d1,d2,off)==0) : true){
                            long long dline = delta_if_swap(a,b,col,scnt,rconf);
                            long long obj = dline + (long long)O.wdiag * ddiag;
                            if(obj < 0 || ddiag < 0){
                                int a_d1=a-col[a]+off,a_d2=a+col[a],b_d1=b-col[b]+off,b_d2=b+col[b];
                                d1[a_d1]--; d2[a_d2]--; d1[b_d1]--; d2[b_d2]--;
                                apply_swap_update(a,b,col,scnt,rconf);
                                d1[a-col[a]+off]++; d2[a+col[a]]++; d1[b-col[b]+off]++; d2[b+col[b]]++;
                                long long nt=0; for(long long v:rconf) nt+=v; row_total=nt;
                                dpp += ddiag;
                                obj_total = row_total + (long long)O.wdiag * dpp;
                                update_global_best(obj_total,col);
                                break;
                            }
                        }
                    }
                };
                shake();
                if(three_cycle_escape(col,d1,d2,off,scnt,rconf,dpp,row_total,obj_total,rng,T,O)){
                    stale=0; continue;
                }
                stale=0;
            }
        }
        if(chrono::steady_clock::now()>deadline) break;
        if(!O.hard_diag && dpp>0){
            for(long long lim=0; lim< (long long)N*4 && dpp>0; ++lim){
                int a=rng()%N, b=rng()%N; if(a==b) continue;
                long long ddiag=diag_pairs_delta_swap(a,b,col,d1,d2,off);
                if(ddiag<0){
                    int a_d1=a-col[a]+off,a_d2=a+col[a],b_d1=b-col[b]+off,b_d2=b+col[b];
                    d1[a_d1]--; d2[a_d2]--; d1[b_d1]--; d2[b_d2]--;
                    apply_swap_update(a,b,col,scnt,rconf);
                    d1[a-col[a]+off]++; d2[a+col[a]]++; d1[b-col[b]+off]++; d2[b+col[b]]++;
                    dpp += ddiag;
                    long long nt=0; for(long long v:rconf) nt+=v; row_total=nt;
                    obj_total = row_total + (long long)O.wdiag * dpp;
                }
            }
            if(dpp>0){ diag_reduce_phase(col,d1,d2,off,scnt,rconf,dpp,row_total,obj_total,rng,T,O); }
        }
        if(check_parallel(col)){
            if(O.verbose){
                auto t1 = chrono::steady_clock::now();
                cerr << "\nsolved, obj=0 dpp=0"
                     << " total_time_s=" << chrono::duration<double>(t1 - t0).count()
                     << " attempt_time_s=" << chrono::duration<double>(t1 - tatt).count()
                     << " attempt=" << attempt
                     << " iters=" << att_iters
                     << "\n";
            }
            cout<<N<<"\n";
            for(int i=0;i<N;++i){ if(i) cout<<' '; cout<<col[i]+1; }
            cout<<"\n";
            if(O.print_time){
                auto t1_now = chrono::steady_clock::now();
                cout << "elapsed_s=" << fixed << setprecision(6)
                     << chrono::duration<double>(t1_now - t0).count() << "\n";
            }
            return 0;
        }
    }
    if(O.verbose){
        auto t1=chrono::steady_clock::now();
        double secs=chrono::duration<double>(t1 - t0).count();
        cerr<<"\nno-solution; attempts="<<MAX_RESTART<<" best_obj="<<global_best<<" elapsed_s="<<secs<<"\n";
    }
    if(O.print_time){
        auto t1=chrono::steady_clock::now();
        cout << "elapsed_s=" << fixed << setprecision(6)
             << chrono::duration<double>(t1 - t0).count() << "\n";
    }
    if(O.dump_best && !global_best_col.empty()){
        cout<<N<<"\n";
        for(int i=0;i<N;++i){ if(i) cout<<' '; cout<<global_best_col[i]+1; }
        cout<<"\n";
    }
    return 1;
}