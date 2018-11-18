#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = Problem::MaxDistance;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].coverRadius << endl;
        //cout << "coverRadius:" << solutions[i].coverRadius << "   bestValue:" << bestValue << endl;
        if (solutions[i].coverRadius >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].coverRadius;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Length obj = output.coverRadius;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ",";
    if (Problem::isTopologicalGraph(input)) {
        log << obj << ",";
    } else {
        auto oldPrecision = log.precision();
        log.precision(2);
        log << fixed << setprecision(2) << (obj / aux.objScale) << ",";
        log.precision(oldPrecision);
    }
    log << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    // EXTEND[szx][2]: save solution in log.
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Distance,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Length &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        TooManyCentersError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::TooManyCentersError) { Log(LogSwitch::Checker) << "TooManyCentersError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    
    //initialize the graph information
    nodeNum = input.graph().nodenum();
    centerNum = input.centernum();
    
    //initialize the tabu information
    facility_tenure.assign(nodeNum, 0);
    user_tenure.assign(nodeNum, 0);
    base_user_tabu_steps_ = centerNum / 10;
    base_facility_tabu_steps_ = (nodeNum - centerNum) / 10;

    aux.adjMat.init(nodeNum, nodeNum);
    fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxDistance);
    for (ID n = 0; n < nodeNum; ++n) { aux.adjMat.at(n, n) = 0; }

    if (Problem::isTopologicalGraph(input)) {
        aux.objScale = Problem::TopologicalGraphObjScale;
        for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
            // only record the last appearance of each edge.
            aux.adjMat.at(e->source(), e->target()) = e->length();
            aux.adjMat.at(e->target(), e->source()) = e->length();
        }

        Timer timer(30s);
        constexpr bool IsUndirectedGraph = true;
        IsUndirectedGraph
            ? Floyd::findAllPairsPaths_symmetric(aux.adjMat)
            : Floyd::findAllPairsPaths_asymmetric(aux.adjMat);
        Log(LogSwitch::Preprocess) << "Floyd takes " << timer.elapsedSeconds() << " seconds." << endl;
    } else { // geometrical graph.
        aux.objScale = Problem::GeometricalGraphObjScale;
        for (ID n = 0; n < nodeNum; ++n) {
            double nx = input.graph().nodes(n).x();
            double ny = input.graph().nodes(n).y();
            for (ID m = 0; m < nodeNum; ++m) {
                if (n == m) { aux.adjMat.at(n, m) = 0; continue; }
                aux.adjMat.at(n, m) = static_cast<Length>(aux.objScale * hypot(
                    nx - input.graph().nodes(m).x(), ny - input.graph().nodes(m).y()));
            }
        }
    }

    aux.coverRadii.init(nodeNum);
    fill(aux.coverRadii.begin(), aux.coverRadii.end(), Problem::MaxDistance);
}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    
    cout << "nodeNum" << nodeNum << "     centerNum" << centerNum << endl;
    // reset solution state.
    bool status = true;
    auto &centers(*sln.mutable_centers());
    centers.Resize(centerNum, Problem::InvalidId);
    // TODO[0]: replace the following random assignment with your own algorithm.
    //Sampling sampler(rand, centerNum);
    //for (ID n = 0; !timer.isTimeOut() && (n < nodeNum); ++n) {
    //    ID center = sampler.replaceIndex();
    //    if (center >= 0) { centers[center] = n; }
    //}
    //
    //sln.coverRadius = 0; // record obj.
    //for (ID n = 0; n < nodeNum; ++n) {
    //    for (auto c = centers.begin(); c != centers.end(); ++c) {
    //        if (aux.adjMat.at(n, *c) < aux.coverRadii[n]) { aux.coverRadii[n] = aux.adjMat.at(n, *c); }
    //    }
    //    if (sln.coverRadius < aux.coverRadii[n]) { sln.coverRadius = aux.coverRadii[n]; }
    //}

    for (int i = 0; i < 1; ++i) {
        facility_tenure.assign(nodeNum, 0);
        user_tenure.assign(nodeNum, 0);
        isServerdNode.assign(nodeNum, false);
        int index = rand.pick(nodeNum);
        dTable.assign(2, vector<int>(nodeNum, INF));
        fTable.assign(2, vector<int>(nodeNum, -1));
        temp_centers.clear();
        temp_centers.push_back(index);
        isServerdNode[index] = true;
        fTable[0].assign(nodeNum, index);
        assign_(dTable[0], index);
        for (int f = 1; f < centerNum; ++f) {//从初始节点开始，依次构造服务节点
            int serverNode = selectNextSeveredNode();
            addNodeToTable(serverNode);
            //cout << "serverNode:  " << serverNode << "   cur_maxLength:" << cur_maxLength << endl;
        }

        int per_count = 0, per_range = 1500;
        for (int t = 0; !timer.isTimeOut() && t <20000; ++t) {
            vector<int> switchNodes = findSeveredNodeNeighbourhood();//待增添节点
            vector<int> switchNodePair = findPair(switchNodes, t);//交换节点对，若存在相同的Mf则随机返回一对
            int f, v;
            if (switchNodePair.size() == 0 || per_count > per_range) {
                while (isServerdNode[f = rand.pick(nodeNum)]);
                v = temp_centers[rand.pick(centerNum)];
                per_count = 0;
            } else {
                f = switchNodePair[0], v = switchNodePair[1];
            }
            per_count++;
            addNodeToTable(f);
            deleteNodeInTable(v);
            if (cur_maxLength < hist_maxLength) {// updtae the history optimal solution
                hist_maxLength = cur_maxLength;
                hist_centers = temp_centers;
                per_count = 0;
                cout << "serverNode:  " << f << "hist_maxLength:" << hist_maxLength << "  cur_maxLength: " << cur_maxLength << "  t: " << i * 1000 + t << endl;
            }
            user_tenure[f] = t + base_user_tabu_steps_ + rand() % 10;
            facility_tenure[v] = t + base_facility_tabu_steps_ + rand() % 100;
        }
    }
    for (int i = 0; i < hist_centers.size(); ++i) {
        centers[i] = hist_centers[i];
    }
    sln.coverRadius = hist_maxLength;
    cout << "maxLength after change: " << hist_maxLength << endl;
    cout << "the result takes: " << timer.elapsedSeconds() << " seconds." << endl;
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}
void Solver::addNodeToTable(int node) {
    isServerdNode[node] = true;
    cur_maxLength = 0;
    temp_centers.push_back(node);
    isServerdNode[node] = true;
    for (int v = 0; v < nodeNum; ++v) {//更新f表和t表
        if (aux.adjMat[node][v] < dTable[0][v]) {
            dTable[1][v] = dTable[0][v];
            dTable[0][v] = aux.adjMat[node][v];
            fTable[1][v] = fTable[0][v];
            fTable[0][v] = node;
        } else if (aux.adjMat[node][v] < dTable[1][v]) {
            dTable[1][v] = aux.adjMat[node][v];
            fTable[1][v] = node;
        }
        if (dTable[0][v] > cur_maxLength)
            cur_maxLength = dTable[0][v];
    }
}
void Solver::deleteNodeInTable(int node) {
    isServerdNode[node] = false;
    cur_maxLength = 0;
    int i = 0;
    for (; i < temp_centers.size() && temp_centers[i] != node; ++i);//寻找要删除的服务节点
    for (; i < temp_centers.size() - 1; ++i) {
        temp_centers[i] = temp_centers[i + 1];
    }
    temp_centers.pop_back();
    for (int v = 0; v < nodeNum; ++v) {
        if (fTable[0][v] == node) {
            fTable[0][v] = fTable[1][v];
            dTable[0][v] = dTable[1][v];
            findNext(v);

        } else if (fTable[1][v] == node) {
            findNext(v);
        }
        if (dTable[0][v] > cur_maxLength)
            cur_maxLength = dTable[0][v];
    }
}
void Solver::findNext(int v) {
    int nextNode = -1, secondLength = INT32_MAX;
    for (int i = 0; i < temp_centers.size(); ++i) {
        int f = temp_centers[i];//寻找下一个次近服务节点
        if (f != fTable[0][v] && aux.adjMat[v][f] < secondLength) {
            secondLength = aux.adjMat[v][f];
            nextNode = f;
        }
    }
    dTable[1][v] = secondLength;
    fTable[1][v] = nextNode;
}
int Solver::selectNextSeveredNode() {
    vector<int> kClosedNode = findSeveredNodeNeighbourhood();
    int serveredNode = kClosedNode[rand.pick(kClosedNode.size())];//从这k个近节点中随机选择一个作为服务节点
    return serveredNode;
}
std::vector<int> Solver::findSeveredNodeNeighbourhood() {
    int maxServerLength = -1;
    vector<int> serveredNodes;
    for (int v = 0; v < nodeNum; ++v) {
        if (dTable[0][v] > maxServerLength) {
            serveredNodes.clear();
            maxServerLength = dTable[0][v];
            serveredNodes.push_back(v);
        } else if (dTable[0][v] == maxServerLength) {
            serveredNodes.push_back(v);
        }
    }
    int serveredNode = serveredNodes[rand.pick(serveredNodes.size())];
    vector<int> kClosedNode; //备选用户节点v的前k个最近节点
    kClosedNode = sortIndexes(serveredNode, kClosed, maxServerLength);
    return kClosedNode;
}
std::vector<int> Solver::sortIndexes(int v, int k, int length) {
    //返回前k个最小值对应的索引值
    vector<int> temporary(nodeNum, -1);
    assign_(temporary, v);
    vector<int> idx(nodeNum);
    vector<int> res;
    for (int i = 0; i != idx.size(); ++i) {
        idx[i] = i;
    }
    sort(idx.begin(), idx.end(),
        [&temporary](int i1, int i2) {return temporary[i1] < temporary[i2]; });
    for (int i = 0; i < nodeNum && i < k; i++) {
        if (isServerdNode[idx[i]] || temporary[idx[i]] >= length) {
            ++k;
            continue;
        }
        //if (temporary[idx[i]] < length)
            res.push_back(idx[i]);
    }
    return res;
}
std::vector<int> Solver::findPair(const std::vector<int>& switchNode, int t)

{
    int tabu_length = INF;//记录禁忌对中最好的函数值
    int no_tabu_length = INF;//记录非禁忌对中最好的函数值
    vector<vector<int>> tabu_res;
    vector<vector<int>> no_tabu_res;
    vector<int> r;
    map<int, int> Mf; //t < serverTableTenure[v] && t < userTableTenure[f]
    for (int i : switchNode) {
        //isServerdNode[i] = true;
        Mf.clear();//存放删除某个服务节点f后产生的最长服务边maxlength，key为f value为maxlength
        for (int j = 0; j < temp_centers.size(); ++j) {
            Mf[temp_centers[j]] = 0;
        }
        for (int v = 0; v < dTable[0].size(); ++v) {
            if (min(aux.adjMat[i][v], dTable[1][v]) > Mf[fTable[0][v]])
                Mf[fTable[0][v]] = min(aux.adjMat[i][v], dTable[1][v]);
        }
        for (int f = 0; f < temp_centers.size(); f++) {
            //选出删除f后所产生的最小最长服务边
            if (facility_tenure[i] > t && user_tenure[temp_centers[f]] > t) {//找出禁忌对中最好的Mf
                if (Mf[temp_centers[f]] == tabu_length) {
                    r.clear();
                    r.push_back(i);
                    r.push_back(temp_centers[f]);
                    r.push_back(tabu_length);
                    tabu_res.push_back(r);
                } else if (Mf[temp_centers[f]] < tabu_length) {
                    tabu_length = Mf[temp_centers[f]];
                    tabu_res.clear();
                    r.clear();
                    r.push_back(i);
                    r.push_back(temp_centers[f]);
                    r.push_back(tabu_length);
                    tabu_res.push_back(r);
                }
            } else {
                if (Mf[temp_centers[f]] == no_tabu_length) {//找出非禁忌对中最好Mf
                    r.clear();
                    r.push_back(i);
                    r.push_back(temp_centers[f]);
                    r.push_back(no_tabu_length);
                    no_tabu_res.push_back(r);
                } else if (Mf[temp_centers[f]] < no_tabu_length) {
                    no_tabu_length = Mf[temp_centers[f]];
                    no_tabu_res.clear();
                    r.clear();
                    r.push_back(i);
                    r.push_back(temp_centers[f]);
                    r.push_back(no_tabu_length);
                    no_tabu_res.push_back(r);
                }
            }
        }
    }
    if (tabu_length < hist_maxLength) {
        if (tabu_res.size() == 0)
            return vector<int>();
        else 
            return tabu_res[rand.pick(tabu_res.size())] ;
    }
    else {
        if (no_tabu_res.size() == 0)
            return vector<int>();
        else
            return no_tabu_res[rand.pick(no_tabu_res.size())];
    }
        

}
void Solver::assign_(std::vector<int>& vec, int adj) {
    for (int i = 0; i < nodeNum; ++i) {
        vec[i] = aux.adjMat[adj][i];
    }
}

#pragma endregion Solver

}
