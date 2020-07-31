const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');

const { User } = require('./models/User');
const { auth } = require('./middleware/auth');
// DB Config
const db = require('./config/keys');

// application/x-www-form-urlenconded
app.use(bodyParser.urlencoded({extended: true}));
// applicaion/json
app.use(bodyParser.json());
app.use(cookieParser());

// Connect to Mongo
const mongoose = require('mongoose');
mongoose.connect(db.mongoURI,{
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(() => console.log('MongoDB connected...'))
.catch(error => console.log(error))


app.get('/', (req, res) => res.send('Develog!'));
app.post('/api/users/register', (req, res) => {
    // 회원 가입시 필요한 정보 client에서 가져와 DB에 넣기
    // body parser를 이용하여 req로 전송
    const user = new User(req.body)

    user.save((error, userInfo) => {
        if(error) return res.json({success: false, error})
        return res.status(200).json({
            success: true
        })
    })
})

app.post('/api/users/login', (req, res) => {
    // 요청된 이메일이 데이터베이스에 있는지 찾기
    User.findOne({ email: req.body.email }, (err, user) => {
        if(!user) {
            return res.json({
                loginSuccess: false,
                message: "User not found."
            })
        }
        // 비밀번호 일치 여부 확인
        user.comparePassword(req.body.password, (err, isMatch) => {
            if(!isMatch)
            return res.json({ 
                loginSuccess: false, 
                message: "Passwords do not match."
            })
            
            // user를 위한 token 생성
            user.generateToken((err, user) => {
                if(err) return res.status(400).send(err);

                // token 저장 (쿠키)
                res.cookie("x_auth", user.token)
                .status(200)
                .json({
                    loginSuccess: true, 
                    userID: user._id})
            })
        })
    })    
})

app.get('/api/users/auth', auth, (req, res)  => {
    // 여기가지 통과 --> authentication이 ture라는 뜻
    res.status(200).json({
        _id: req.user._id,
        isAdmin: req.user.role === 0 ? false : true,
        isAuth: true, 
        email: req.user.email,
        name: req.user.name,
        lastname: req.user.lastname,
        role: req.user.role,
        image: req.user.image
    })
})


const PORT = process.env.PORT || 3000
app.listen(PORT, () => console.log(`Example app listening on port ${PORT}!`))