#！usr/bin/bash
#push branch to github branch
#branchName=`git rev-parse --abbrev-ref HEAD`
#echo "git branch:$branchName"
git add --all .
message=`date +%Y-%m-%d-%H:%M`
while getopts m: opt
do
    case $opt in
        m)
            message=$OPTARG
            ;;
        ?)
            echo "Usage:args [-m]"
            echo "-m means message"
            echo "exit"
            exit
            ;;
    esac
done
git commit -m "$message"
#git fetch github
#git merge --no-edit github/master
git commit -m "merge $message"
#remote=`git remote`
#echo "push to:$remote"
git push github $branchName
