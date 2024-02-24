---
layout: post
title: Remote Server Cheatsheet
date: 2023-04-20
description: Cheatsheet of Jupyter, tmux, etc.
tags: ["python", "tmux"]
categories: codes
# toc:
#     sidebar: left
---

## Jupyter Notebook

1. Login your remote server.

   ```ssh
   ssh username@server
   ```

2. Create a remote Jupyter notebook on the server.

   ```powershell
   jupyter notebook --no-browser --port=8080
   ```

   Alternatively, you can define a macro in your configuration file `.bashrc` on the remote server.

   ```powershell
   function nb(){
       jupyter notebook --no-browser --port=8080
    }
   ```

3. Open a new local terminal and connect to the remote Jupyter notebook

   ```ssh
   ssh -L 8080:localhost:8080 username@server
   ```

4. Open <https://localhost:8080> in your browser.

## SSH

It is quite tedious to repeat above commands every time.
Here's how you can configure your own SSH profile to make it easier to connect to your remote server:

1. Add the following profile to your `.ssh/config` file.

   ```ssh
   Host myhost
       Hostname server
       User username
       Port myport
       LocalForward 8080 localhost:8080
   ```

2. Connect and listen to the remote server simply through `ssh myhost`.

## Tmux

Tmux is a "terminal multiplexer", it enables a number of terminals (or windows) to be accessed and controlled from a single terminal.

1. Install [Tmux Plugin Manager](https://github.com/tmux-plugins/tpm) and [Tmux Resurrect](https://github.com/tmux-plugins/tmux-resurrect).
2. Start a new session with the name _mysession_.

   ```powershell
   tmux new -s mysession
   ```

3. Save a session by `Ctrl-b + Ctrl-s` and restore a session by `Ctrl-b + Crtl-r`.
4. Show all sessions.

   ```powershell
   tmux list-session
   ```

5. Kill session _mysession_.

   ```powershell
   tmux kill-session -t mysession
   ```
