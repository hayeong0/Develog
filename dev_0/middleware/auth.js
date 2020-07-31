const { User } = require('../models/User');

let auth = (req, res, next) => {
    //인증 처리 하는 곳

    // client 쿠키에서 토큰을 가져온다
    let token = req.cookies.x_auth;

    // 토큰을 decode, 유저를 찾는다
    User.findeByToken(token, (err, user) => {
        if(err) throw err;
        if(!user) return res.json({ isAuth: false, error: true});
        
        req.token = token;
        req.user = user;
        // middle ware에서 갈 수 있게끔
        next();
    })

    // 유저가 있으면 O, 없으면 인증 x
}

module.exports = {auth}; 