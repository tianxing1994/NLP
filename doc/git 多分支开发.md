### git 多分支开发

创建新的分支
```text
git checkout -b new_branch
```

查看现有分支
```text
git branch
```

示例: 
```text
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git checkout -b desktop_computer  
Switched to a new branch 'desktop_computer'  
M       .idea/workspace.xml  
  
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git branch  
* desktop_computer  
  master  
  
(NLP) D:\Users\Administrator\PycharmProjects\NLP>  
```

将本地分支推送到远程仓库对应分支: 
```text
# 查看当前分支, 确认目前在自己的工作分支. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git branch  
* desktop_computer  
  master 
  
# git add, commit 保存当前修改. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git add .  
......
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git commit -m "commit changed"   
......
  
# 切换到 master 分支. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git checkout master  
Switched to branch 'master'
Your branch is up to date with 'origin/master'. 
  
# git pull. 将 master 分支更新为与远程仓库相同. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git pull
Already up to date.
  
# 切换到 desktop_computer 工作分支
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git checkout desktop_computer
Switched to branch 'desktop_computer'
  
# git rebase, 更新 desktop_computer 工作分支中与 master 分支不同的部分, 使不同的部分的修改起点为当前 master 分支的最新节点. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git rebase master
Current branch desktop_computer is up to date.
  
# 将本地的 desktop_computer 分支推送到远程仓库的 desktop_computer 分支. 
(NLP) D:\Users\Administrator\PycharmProjects\NLP>git push origin desktop_computer:desktop_computer
Enumerating objects: 24, done.
Counting objects: 100% (24/24), done.
......
  
# 接下来去远程仓库查看新提交的 desktop_computer 分支, 并将其合入 master 分支. 
# 在远程仓库创建 "Pull requests". 等等, 我还不甚清楚明白. 
```

